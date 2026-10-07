from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    text: bool = True
    images: bool = False


TEXT_ONLY_CAPABILITIES = ProviderCapabilities()
IMAGE_CAPABILITIES = ProviderCapabilities(images=True)


def messages_contain_images(
    messages: Sequence[Mapping[str, object]],
) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "image_url":
                return True
    return False

