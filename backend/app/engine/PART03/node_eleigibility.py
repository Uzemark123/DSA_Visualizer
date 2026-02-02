from __future__ import annotations

from typing import Optional


def node_chunk_tier(node: object) -> Optional[int]:
    """
    Extract the chunk tier from a roadmap node.
    Supports both `.chunk_pointer` (your Node model) and `.chunk` (possible legacy shape).
    """
    chunk = getattr(node, "chunk_pointer", None) or getattr(node, "chunk", None)
    if chunk is None:
        return None
    return getattr(chunk, "tier", None)


__all__ = ["node_chunk_tier"]
