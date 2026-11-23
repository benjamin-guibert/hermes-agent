"""Discord native media uploads and delivery fallbacks."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from functools import wraps

from gateway.platforms.event import current_reply_trigger
from gateway.outbound.operation import outbound_operation, outbound_delivery_scope
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.i18n import t
from gateway.platforms.base import SendResult
from plugins.platforms.discord.archived_redirect import mark_redirect_delivered, redirect_archived_send

logger = logging.getLogger("plugins.platforms.discord.adapter")

# Default Discord attachment cap for DMs / channels without a guild boost
# context. 20 MiB since the Sep 3 2026 API change (10 MiB before); guild
# channels expose a boost-raised limit via ``guild.filesize_limit``, but
# discord.py's fallback constant can lag the platform default, so the
# effective limit is never taken below this floor. See issue #50846 and
# https://docs.discord.com/developers/change-log (Sep 3, 2026).
_DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES = 20 * 1024 * 1024
# A long reply file stays under the smallest attachment cap Discord has had (8 MiB), so it never trips
# the per-channel upload preflight; a bigger reply goes out as several files.
_DISCORD_REPLY_FILE_MAX_BYTES = 7 * 1024 * 1024


_OPUS_GRANULE_RATE = 48000  # Ogg/Opus granule positions always count 48 kHz samples (RFC 7845 §4)


def _ogg_opus_duration(data: bytes) -> Optional[float]:
    """Playback seconds of an Ogg/Opus stream (last page's granule position minus the OpusHead
    pre-skip), or None when ``data`` is not Ogg/Opus. Dependency-free: Discord needs the real
    length for the voice-message scrubber, and a byte-size guess is wrong for every bitrate."""
    if data[:4] != b"OggS" or len(data) < 27:
        return None
    head = 27 + data[26]  # first packet follows the page header and its segment table
    if data[head:head + 8] != b"OpusHead" or len(data) < head + 12:
        return None
    pre_skip = int.from_bytes(data[head + 10:head + 12], "little")
    granule, pos = None, 0
    # Walk page headers by their segment tables: an "OggS" byte run inside packet data is not a page.
    while data[pos:pos + 4] == b"OggS" and pos + 27 <= len(data):
        body_start = pos + 27 + data[pos + 26]
        if body_start > len(data):
            break
        page_granule = int.from_bytes(data[pos + 6:pos + 14], "little", signed=True)
        if page_granule != -1:  # -1: no packet ends on this page
            granule = page_granule
        pos = body_start + sum(data[pos + 27:body_start])
    if granule is None:
        return None
    secs = (granule - pre_skip) / _OPUS_GRANULE_RATE
    return secs if secs > 0 else None


def media_operation(method):
    @wraps(method)
    async def send(self, *args, **kwargs):
        scope = current_reply_trigger.get()
        in_final = isinstance(scope, tuple) and len(scope) == 4 and scope[1] is asyncio.current_task()
        if outbound_operation.get() is not None or in_final:
            return await method(self, *args, **kwargs)
        with outbound_delivery_scope():
            return await method(self, *args, **kwargs)
    return send


def media_delivery_result(channel, success, ids, error=None, *, ambiguous=False):
    from plugins.platforms.discord.adapter import discord
    raw = {"message_ids": list(ids), "ambiguous": ambiguous, "prevent_fallback": ambiguous}
    if channel is not None:
        raw.update(chat_id=str(channel.id), thread_id=str(channel.id) if isinstance(channel, discord.Thread) else None)
    if not success and ids:
        raw["partial_overflow"] = True
    return SendResult(success=success, message_id=(ids[-1] if success else ids[0]) if ids else None, error=error, raw_response=raw)


async def fallback_image_chunk(adapter, channel, chunk, metadata):
    from urllib.request import url2pathname
    ids = []
    for image_url, alt_text in chunk:
        if image_url.startswith("file://"):
            result = await adapter.send_image_file(channel.id, url2pathname(image_url[7:]), alt_text, metadata=metadata)
        elif adapter._is_animation_url(image_url):
            result = await adapter.send_animation(channel.id, image_url, alt_text, metadata=metadata)
        else:
            result = await adapter.send_image(channel.id, image_url, alt_text, metadata=metadata)
        raw = result.raw_response or {}
        ids.extend(raw.get("message_ids") or ([result.message_id] if result.message_id else []))
        if not result.success:
            return media_delivery_result(channel, False, ids, result.error, ambiguous=bool(raw.get("ambiguous")))
    return media_delivery_result(channel, True, ids)


async def resolve_image_route(adapter, chat_id, metadata):
    from plugins.platforms.discord.adapter import _prompt_target_id
    try:
        channel = await adapter._resolve_channel(_prompt_target_id(chat_id, metadata))
        if channel is None:
            raise ValueError("Discord image channel not found")
        channel, prefix, redirected = await redirect_archived_send(
            adapter, channel, "", (metadata or {}).get("reply_to_message_id"), metadata)
        return channel, prefix, redirected, None
    except Exception as exc:
        logger.debug("Discord image route failed", exc_info=True)
        return None, "", False, SendResult(success=False, error=str(exc))


async def image_batch_failure(adapter, channel, chunk, metadata, confirmed_ids, error):
    if confirmed_ids or getattr(error, "status", None) not in (400, 403, 404, 413, 415, 422):
        return media_delivery_result(channel, False, confirmed_ids, str(error) or type(error).__name__,
                                     ambiguous=getattr(error, "status", None) is None)
    logger.debug("Discord image batch rejected; retrying individual images", exc_info=True)
    return await fallback_image_chunk(adapter, channel, chunk, metadata)


def image_batch_caption(adapter, captions, notices, prefix):
    content = captions[0] if captions else None
    if prefix:
        content = prefix + (content or "")
    if notices:
        content = adapter.warning_text("\n".join(([content] if content else []) + notices), content)
    return content


class DiscordMediaMixin:
    REPLY_FILE_MAX_BYTES = _DISCORD_REPLY_FILE_MAX_BYTES

    def reply_message_count(self, content: str, metadata: Optional[dict[str, Any]] = None) -> int:
        """Messages ``send(content)`` would post (its 2,000-char split of the formatted text)."""
        formatted = self.format_message(content)
        return 1 if len(formatted) <= self.MAX_MESSAGE_LENGTH else len(self.truncate_message(formatted, self.MAX_MESSAGE_LENGTH))

    async def send_reply_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a long reply file (#152012). Unlike ``send_document``, a message Discord accepted
        without the attachment is deleted again, so a failed upload leaves nothing in the channel and
        the caller can send the reply as messages instead."""
        result = await self._send_local_file(chat_id, file_path, caption, file_name=file_name,
                                             not_found="File not found", kind="reply file", metadata=metadata)
        if not result.success and result.message_id:  # accepted without the attachment (#66797)
            actual_chat = (result.raw_response or {}).get("chat_id", chat_id)
            if await self.delete_message(actual_chat, str(result.message_id)):
                result.raw_response = {**(result.raw_response or {}),
                                       "deleted_message_ids": [str(result.message_id)], "message_ids": []}
                result.message_id = None
        return result

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message the bot posted; False when Discord refuses (missing permission, gone)."""
        from plugins.platforms.discord.adapter import _prompt_target_id
        if not self._client:
            return False
        try:
            channel = await self._resolve_channel(_prompt_target_id(chat_id, None))
            if channel is None:
                return False
            await channel.get_partial_message(int(message_id)).delete()
            return True
        except Exception as e:  # health: allow BLE001 -- best-effort delete: every failure is a logged False
            logger.debug("[%s] Could not delete Discord message %s: %s", self.name, message_id, e)
            return False

    @staticmethod
    def _discord_upload_limit_bytes(channel: Any) -> int:
        """Return the effective Discord attachment size limit for *channel*.

        Prefer the guild's boost-aware ``filesize_limit`` when present; fall
        back to the platform default for DMs / group DMs without a guild.
        The guild value is floored at the platform default: discord.py's
        unboosted-tier constant can lag a platform-wide raise (10 MiB in
        2.7.1 vs the 20 MiB default since Sep 3 2026), and under-reporting
        makes the preflight reject files Discord would accept.
        """
        guild = getattr(channel, "guild", None)
        if guild is not None:
            limit = getattr(guild, "filesize_limit", None)
            if isinstance(limit, int) and limit > 0:
                return max(limit, _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES)
        return _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    async def _reject_oversized_upload(
        self, channel: Any, file_path: str, filename: str, *, caption: Optional[str] = None,
    ) -> Optional[SendResult]:
        """Preflight ``file_path`` against the channel's upload cap (#50846): a doomed
        ``413`` round-trip is skipped and the user gets a notice naming the size and the
        limit. Returns the failed result, or ``None`` when the file may be uploaded."""
        try:
            file_size = os.path.getsize(file_path)
        except OSError as exc:
            return SendResult(success=False, error=f"Cannot stat file {filename}: {exc}")
        limit = self._discord_upload_limit_bytes(channel)
        if file_size <= limit:
            return None
        size_mb = file_size / (1024 * 1024)
        limit_mb = limit / (1024 * 1024)
        error = f"File too large for Discord upload: {filename} is {size_mb:.1f} MB (limit {limit_mb:.0f} MB)"
        logger.warning("[%s] %s", self.name, error)
        notice = t(
            "platform.discord.media.upload_too_large",
            filename=filename, size_mb=f"{size_mb:.1f}", limit_mb=f"{limit_mb:.0f}")
        try:
            shown = self.warning_notifications_enabled()
            if shown:
                # Legacy: the notice is posted in-channel only (forum parents get no bare notice).
                if not self._is_forum_parent(channel):
                    await channel.send(content=notice)
            elif caption:
                # Hidden: the requested caption still travels, on the channel's native shape.
                if self._is_forum_parent(channel):
                    await self._send_to_forum(channel, caption)
                else:
                    await channel.send(content=caption)
        except Exception:
            logger.debug("[%s] Failed to send oversized-file notice for %s", self.name, filename, exc_info=True)
        return SendResult(success=False, error=error)

    @media_operation
    async def _send_file_attachment(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local file as a Discord attachment (forum channels get a new thread). Path-based
        ``discord.File`` only: the open-handle form can race the multipart encoder after an image
        batch and yield zero attachments — a silent drop for video/document MEDIA tags.

        See #66797.
        """
        from plugins.platforms.discord.adapter import _prompt_target_id, discord

        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not os.path.isfile(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")
        channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
        if not channel:
            return SendResult(success=False, error=f"Channel {chat_id} not found")
        channel, caption, redirected = await redirect_archived_send(
            self, channel, caption or "", (metadata or {}).get("reply_to_message_id"), metadata)
        filename = file_name or os.path.basename(file_path)
        rejected = await self._reject_oversized_upload(channel, file_path, filename, caption=caption)
        if rejected is not None:
            return rejected
        logger.info(
            "[%s] Sending file attachment %s (%s) to %s", self.name, filename,
            os.path.splitext(filename)[1].lower() or "no-ext", chat_id,
        )
        # Path-based File (discord.py owns open/close); ``files=[...]`` over deprecated ``file=``.
        discord_file = discord.File(file_path, filename=filename)
        if self._is_forum_parent(channel):
            result = await self._forum_post_file(
                channel, content=(caption or "").strip(), files=[discord_file],
            )
            return result
        try:
            msg = await channel.send(content=caption if caption else None, files=[discord_file])
        except Exception as exc:
            logger.debug("Discord attachment POST failed", exc_info=True)
            return media_delivery_result(channel, False, [], str(exc) or type(exc).__name__,
                                         ambiguous=getattr(exc, "status", None) is None)
        attachments = getattr(msg, "attachments", None) or []
        if not attachments:
            # Discord accepted the message but attached nothing: fail loud instead of a silent drop.
            # Discord accepted the message but attached nothing — the failure mode reported in #66797 (MEDIA
            # video stripped from text, no attachment, no prior log line).
            logger.warning(
                "[%s] Discord returned message %s with no attachments for %s", self.name,
                getattr(msg, "id", "?"), filename,
            )
            return SendResult(
                success=False,
                error=f"Discord accepted the message but attached no files ({filename})",
                message_id=str(getattr(msg, "id", "") or "") or None,
                raw_response={"chat_id": str(channel.id), "thread_id": str(channel.id) if isinstance(channel, discord.Thread) else None, "message_ids": [str(msg.id)]},
            )
        if redirected:
            mark_redirect_delivered()
        return SendResult(success=True, message_id=str(msg.id), raw_response={
            "chat_id": str(channel.id), "thread_id": str(channel.id) if isinstance(channel, discord.Thread) else None, "message_ids": [str(msg.id)]})


    @media_operation
    async def send_multiple_images(
        self, chat_id: str, images: list[tuple[str, str]],
        metadata: Optional[dict[str, Any]] = None, human_delay: float = 0.0,
    ) -> SendResult:
        """Send images as one Discord message (<=10 attachments): URLs are downloaded and uploaded
        inline (bare links don't render); on chunk failure the remainder uses the per-image loop."""
        from plugins.platforms.discord.adapter import _prompt_target_id, _image_ext_from_content_type, _read_url_image_with_redirect_guard, is_safe_url

        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not images:
            return SendResult(success=False, error="no images to send")
        try:
            import discord as _discord_mod
            import io as _io
            from urllib.parse import unquote as _unquote
        except Exception:  # pragma: no cover
            return await super().send_multiple_images(chat_id, images, metadata, human_delay)
        channel, redirect_prefix, redirected, route_error = await resolve_image_route(self, chat_id, metadata)
        if route_error is not None:
            return route_error
        CHUNK = 10
        chunks = [images[i:i + CHUNK] for i in range(0, len(images), CHUNK)]
        delivered = False
        message_ids = []
        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)
            files: list[Any] = []
            captions: list[str] = []
            skip_notices: list[str] = []
            aiohttp_session = None
            try:
                for image_url, alt_text in chunk:
                    if alt_text:
                        captions.append(alt_text)
                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        if not os.path.exists(local_path):
                            logger.warning("[%s] Skipping missing image: %s", self.name, local_path)
                            continue
                        # Same preflight as _send_file_attachment (#50846): an oversized
                        # local image would 413 the whole chunk and drop its siblings
                        # into the fallback path.
                        try:
                            _img_size = os.path.getsize(local_path)
                        except OSError as stat_err:
                            logger.warning(
                                "[%s] Skipping unreadable image %s: %s",
                                self.name, local_path, stat_err,
                            )
                            continue
                        _img_limit = self._discord_upload_limit_bytes(channel)
                        if _img_size > _img_limit:
                            logger.warning(
                                "[%s] Skipping oversized image in batch: %s is %.1f MB (limit %.0f MB)",
                                self.name, os.path.basename(local_path),
                                _img_size / (1024 * 1024), _img_limit / (1024 * 1024),
                            )
                            skip_notices.append(t(
                                "platform.discord.media.image_skipped_too_large",
                                filename=os.path.basename(local_path),
                                size_mb=f"{_img_size / (1024 * 1024):.1f}",
                                limit_mb=f"{_img_limit / (1024 * 1024):.0f}"))
                            continue
                        files.append(_discord_mod.File(local_path, filename=os.path.basename(local_path)))
                    else:
                        if not is_safe_url(image_url):
                            logger.warning("[%s] Blocked unsafe image URL in batch", self.name)
                            continue
                        # Download to BytesIO so it renders inline
                        try:
                            import aiohttp as _aiohttp
                            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
                            _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
                            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)
                            if aiohttp_session is None:
                                aiohttp_session = _aiohttp.ClientSession(**_sess_kw)
                            status, data, headers = await _read_url_image_with_redirect_guard(
                                aiohttp_session, image_url,
                                timeout=_aiohttp.ClientTimeout(total=30), request_kwargs=_req_kw,
                            )
                            if status != 200:
                                logger.warning(
                                    "[%s] Failed to download image (HTTP %d) in batch: %s",
                                    self.name, status, image_url[:80],
                                )
                                continue
                            ext = _image_ext_from_content_type(headers.get("content-type", "image/png"))
                            files.append(_discord_mod.File(_io.BytesIO(data), filename=f"image_{len(files)}.{ext}"))
                        except Exception as dl_err:
                            logger.warning("[%s] Download failed for %s: %s", self.name, image_url[:80], dl_err)
                            continue
                if not files:
                    # Everything in this chunk was skipped. Still surface any
                    # oversized-file notices so the drop is not silent.
                    shown = self.warning_notifications_enabled()
                    if not shown and captions:
                        if self._is_forum_parent(channel):
                            await self._send_to_forum(channel, captions[0])
                        else:
                            await channel.send(content=captions[0])
                    if skip_notices and shown and not self._is_forum_parent(channel):
                        try:
                            await channel.send(content="\n".join(skip_notices))
                        except Exception:
                            logger.debug(
                                "[%s] Failed to send oversized-image notices",
                                self.name, exc_info=True,
                            )
                    continue
                # Use the first caption if any (Discord only has one message body for the group)
                content = image_batch_caption(self, captions, skip_notices, redirect_prefix)
                logger.info(
                    "[%s] Sending %d image(s) as single Discord message (chunk %d/%d)",
                    self.name, len(files), chunk_idx + 1, len(chunks),
                )
                if self._is_forum_parent(channel):
                    await self._forum_post_file(
                        channel, content=(content or "").strip(), files=files,
                    )
                else:
                    message = await channel.send(content=content, files=files)
                    message_ids.append(str(message.id))
                delivered = True
                if redirected:
                    mark_redirect_delivered()
                    redirect_prefix = ""
            except Exception as e:
                fallback = await image_batch_failure(self, channel, chunk, metadata, message_ids, e)
                if not fallback.success:
                    return fallback
                message_ids.extend((fallback.raw_response or {}).get("message_ids", []))
                delivered = True
            finally:
                if aiohttp_session is not None:
                    try:
                        await aiohttp_session.close()
                    except Exception:
                        pass
        return media_delivery_result(channel, delivered, message_ids, None if delivered else "all images failed to send")


    @media_operation
    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
        is_voice: Optional[bool] = None, **kwargs,
    ) -> SendResult:
        """Keep native voice mode; only a definitive native rejection permits a file fallback."""
        if is_voice is False:
            return await self._send_local_file(chat_id, audio_path, caption,
                not_found="Audio file not found", kind="audio", metadata=metadata)
        from plugins.platforms.discord.adapter import _prompt_target_id, discord
        if not await asyncio.to_thread(os.path.isfile, audio_path):
            return SendResult(success=False, error=f"Audio file not found: {audio_path}")
        channel = None
        ids = []
        try:
            import io
            channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            if await asyncio.to_thread(os.path.getsize, audio_path) > self._discord_upload_limit_bytes(channel):
                channel, _, invalid_redirect = await redirect_archived_send(
                    self, channel, "", reply_to or (metadata or {}).get("reply_to_message_id"), metadata)
                if not invalid_redirect:
                    return await self._reject_oversized_upload(channel, audio_path, os.path.basename(audio_path), caption=caption)
                return media_delivery_result(channel, False, [], "Audio file too large for Discord upload")
            voice = await self._ogg_opus_voice_payload(audio_path)
            channel, routed_caption, redirected = await redirect_archived_send(
                self, channel, caption or "", reply_to or (metadata or {}).get("reply_to_message_id"), metadata)
            if redirected:
                reply_to = None
                if routed_caption:
                    prefix = await channel.send(content=routed_caption)
                    ids.append(str(prefix.id))
                    mark_redirect_delivered()
            reference = self._reply_reference_for_send(reply_to, channel)
            if self._is_forum_parent(channel):
                forum_file = discord.File(io.BytesIO(await asyncio.to_thread(Path(audio_path).read_bytes)), filename=os.path.basename(audio_path))
                return await self._forum_post_file(channel, content=(caption or "").strip(), file=forum_file)
            if voice is not None:
                try:
                    result = await self._post_voice_message(channel, *voice, reply_to=reply_to, reference=reference)
                    raw = result.raw_response or {}
                    result.raw_response = {**raw, "message_ids": [*ids, *(raw.get("message_ids") or [])]}
                    return result
                except Exception as exc:
                    logger.debug("Discord native voice POST failed", exc_info=True)
                    if getattr(exc, "status", None) not in (400, 403, 404, 413, 415, 422):
                        return media_delivery_result(channel, False, ids, str(exc) or type(exc).__name__, ambiguous=True)
            file_data = await asyncio.to_thread(Path(audio_path).read_bytes)
            file = discord.File(io.BytesIO(file_data), filename=os.path.basename(audio_path))
            try:
                message = await channel.send(file=file, reference=reference)
            except Exception as exc:
                if reference is None or not self._is_reply_reference_rejected(exc):
                    raise
                message = await channel.send(file=file, reference=None)
            ids.append(str(message.id))
            return media_delivery_result(channel, True, ids)
        except Exception as exc:
            logger.debug("Discord voice delivery failed", exc_info=True)
            return media_delivery_result(channel, False, ids, str(exc) or type(exc).__name__, ambiguous=bool(channel))

    @staticmethod
    async def _ogg_opus_voice_payload(audio_path: str) -> Optional[tuple[bytes, float]]:
        """``(ogg_opus_bytes, duration_secs)`` for a voice message: the file itself when it already
        is Ogg/Opus, else an ffmpeg transcode; None when neither yields Ogg/Opus."""
        from gateway.platforms.base import transcode_to_ogg_opus

        data = await asyncio.to_thread(Path(audio_path).read_bytes)
        duration = _ogg_opus_duration(data)
        if duration is not None:
            return data, duration
        converted = await asyncio.to_thread(transcode_to_ogg_opus, audio_path, bitrate="48k", timeout=30)
        if not converted:
            return None
        try:
            data = await asyncio.to_thread(Path(converted).read_bytes)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(converted)
        duration = _ogg_opus_duration(data)
        return (data, duration) if duration is not None else None

    async def _post_voice_message(
        self, channel: Any, data: bytes, duration_secs: float, *, reply_to: Optional[str], reference: Any,
    ) -> SendResult:
        """POST a native voice message (flags=8192) with Ogg/Opus ``data`` via the raw API."""
        from plugins.platforms.discord.adapter import discord

        payload_data: dict[str, Any] = {
            "flags": 8192,
            "attachments": [{
                "id": "0", "filename": "voice-message.ogg", "duration_secs": round(duration_secs, 2),
                "waveform": base64.b64encode(bytes([128] * 256)).decode(),
            }],
        }
        if reference is not None:
            payload_data["message_reference"] = {"message_id": str(reply_to), "fail_if_not_exists": False}
        form = [
            {"name": "payload_json", "value": json.dumps(payload_data)},
            {"name": "files[0]", "value": data, "filename": "voice-message.ogg", "content_type": "audio/ogg"},
        ]
        msg_data = await self._client.http.request(
            discord.http.Route("POST", "/channels/{channel_id}/messages", channel_id=channel.id),
            form=form,
        )
        return SendResult(success=True, message_id=str(msg_data["id"]), raw_response={
            "chat_id": str(channel.id), "thread_id": str(channel.id) if isinstance(channel, discord.Thread) else None, "message_ids": [str(msg_data["id"])]})


    async def _send_local_file(self, chat_id, path, caption, *, file_name=None, not_found: str, kind: str, metadata=None):
        """A failed attachment is a failed delivery, never a successful text notice."""
        try:
            return await self._send_file_attachment(chat_id, path, caption, file_name=file_name, metadata=metadata)
        except FileNotFoundError:
            return SendResult(success=False, error=f"{not_found}: {path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send %s: %s", self.name, kind, e, exc_info=True)
            return SendResult(success=False, error=str(e))


    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local image file natively as a Discord file attachment."""
        return await self._send_local_file(
            chat_id, image_path, caption, not_found="Image file not found", kind="local image",
            metadata=metadata,
        )


    @media_operation
    async def _send_url_media(
        self, chat_id: str, url: str, caption: Optional[str], *, kind: str,
        filename_for, fallback, metadata: Optional[dict], error_metadata: Optional[dict],
    ) -> SendResult:
        """Download ``url`` and post it as a native attachment (Discord renders those inline).
        ``fallback(metadata)`` is the base-adapter URL send (``error_metadata`` after download failure)."""
        from plugins.platforms.discord.adapter import _prompt_target_id, _read_url_image_with_redirect_guard, discord, is_safe_url

        if not self._client:
            return SendResult(success=False, error="Not connected")
        if not is_safe_url(url):
            logger.warning("[%s] Blocked unsafe %s URL during Discord send_%s", self.name, kind, kind)
            return await fallback(metadata)
        post_started = False
        channel = None
        try:
            import aiohttp
            channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")
            channel, caption, redirected = await redirect_archived_send(
                self, channel, caption or "", (metadata or {}).get("reply_to_message_id"), metadata)
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
            _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(resolve_proxy_url(platform_env_var="DISCORD_PROXY"))
            async with aiohttp.ClientSession(**_sess_kw) as session:
                status, data, headers = await _read_url_image_with_redirect_guard(
                    session, url, timeout=aiohttp.ClientTimeout(total=30), request_kwargs=_req_kw,
                )
                if status != 200:
                    raise Exception(f"Failed to download {kind}: HTTP {status}")
                import io
                file = discord.File(io.BytesIO(data), filename=filename_for(headers))
                if self._is_forum_parent(channel):
                    return await self._forum_post_file(channel, content=(caption or "").strip(), file=file)
                post_started = True
                msg = await channel.send(content=caption if caption else None, file=file)
                if redirected:
                    mark_redirect_delivered()
                return SendResult(success=True, message_id=str(msg.id), raw_response={"chat_id": str(channel.id), "thread_id": str(channel.id) if isinstance(channel, discord.Thread) else None, "message_ids": [str(msg.id)]})
        except ImportError:
            logger.warning("[%s] aiohttp not installed, falling back to URL. Run: pip install aiohttp", self.name, exc_info=True)
            return await fallback(error_metadata)
        except Exception as e:  # pragma: no cover - defensive logging
            if post_started and getattr(e, "status", None) not in (400, 403, 404, 413, 415, 422):
                return media_delivery_result(channel, False, [], str(e) or type(e).__name__, ambiguous=True)
            logger.error("[%s] Failed to send %s attachment, falling back to URL: %s", self.name, kind, e, exc_info=True)
            return await fallback(error_metadata)


    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image natively as a Discord file attachment."""
        from plugins.platforms.discord.adapter import _prompt_target_id, _image_ext_from_content_type

        return await self._send_url_media(
            chat_id, image_url, caption, kind="image",
            filename_for=lambda h: f"image.{_image_ext_from_content_type(h.get('content-type', 'image/png'))}",
            fallback=lambda md: super(DiscordMediaMixin, self).send_image(chat_id, image_url, caption, reply_to, metadata=md),
            metadata=metadata, error_metadata=metadata,
        )


    async def send_animation(
        self, chat_id: str, animation_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send an animated GIF natively as a Discord file attachment."""
        return await self._send_url_media(
            chat_id, animation_url, caption, kind="animation", filename_for=lambda _h: "animation.gif",
            fallback=lambda md: super(DiscordMediaMixin, self).send_animation(chat_id, animation_url, caption, reply_to, metadata=md),
            metadata=metadata, error_metadata=metadata,
        )


    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local video file natively as a Discord attachment."""
        return await self._send_local_file(
            chat_id, video_path, caption, not_found="Video file not found", kind="local video",
            metadata=metadata,
        )


    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Send an arbitrary file natively as a Discord attachment."""
        return await self._send_local_file(
            chat_id, file_path, caption, file_name=file_name, not_found="File not found", kind="document",
            metadata=metadata,
        )

