"""Operation-local routing state for outbound text and media.

Plugins read ``outbound_operation`` for a cache shared by one logical send.
``outbound_delivery_scope`` creates a fresh cache and removes human reply
permission. Call it at an external delivery boundary, not for each chunk.
The state is task-local and must not be retained after the operation.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from gateway.platforms.event import current_reply_trigger

outbound_operation = ContextVar("outbound_operation", default=None)


@contextmanager
def outbound_delivery_scope():
    token = outbound_operation.set({})
    human = current_reply_trigger.set(None)
    try:
        yield
    finally:
        current_reply_trigger.reset(human)
        outbound_operation.reset(token)


def outbound_delivery(method):
    @wraps(method)
    def deliver(*args, **kwargs):
        with outbound_delivery_scope():
            return method(*args, **kwargs)
    return deliver
