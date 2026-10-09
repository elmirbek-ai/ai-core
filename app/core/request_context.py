from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    method: str
    route: str


request_context: ContextVar[RequestContext | None] = ContextVar(
    "ai_core_request_context", default=None
)


def resolve_request_id(value: bytes | None) -> str:
    # An intentionally narrow subset of printable ASCII prevents log injection.
    if value and len(value) <= 128:
        allowed = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._:"
        if all(character in allowed for character in value):
            return value.decode("ascii")
    return str(uuid4())
