"""Label regime detection (opaque vs. semantic) and opacification utilities."""

from __future__ import annotations

import re
import uuid

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_HEX_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{8,}$")
_OPAQUE_CODE_RE = re.compile(r"^(?:c|cls|class|id|lbl|label)?[-_]?[0-9]+$", re.IGNORECASE)


def is_opaque_label(label: str) -> bool:
    """Return True if a label appears to be an opaque identifier rather than semantic text."""
    cleaned = label.strip()
    if not cleaned:
        return True
    if cleaned.isdigit():
        return True
    if _UUID_RE.match(cleaned) or _HEX_RE.match(cleaned):
        return True
    if _OPAQUE_CODE_RE.match(cleaned):
        return True
    return False


def detect_opaque_labels(labels: list[str]) -> bool:
    """Return True if the label set is predominantly opaque (e.g., numeric/ID codes)."""
    if not labels:
        return False
    opaque_count = sum(1 for lbl in labels if is_opaque_label(lbl))
    return opaque_count >= max(1, len(labels) // 2)


def make_opaque_mapping(
    labels: list[str],
    mode: str = "uuid",
    namespace: str = "defclass",
) -> dict[str, str]:
    """Map each original label to a unique opaque display identifier."""
    if mode == "uuid":
        return {
            lbl: str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}:{lbl}"))
            for lbl in labels
        }
    if mode == "numbered":
        return {lbl: str(idx) for idx, lbl in enumerate(labels, start=1)}
    raise ValueError(f"Unsupported opaque_mode {mode!r}. Expected 'uuid' or 'numbered'.")


def match_allowed_label(raw_label: str, allowed_labels: list[str]) -> str:
    """Normalize a model string output to the closest valid candidate label."""
    cleaned = raw_label.strip().strip("\"'")
    if cleaned in allowed_labels:
        return cleaned
    lower_map = {lbl.lower(): lbl for lbl in allowed_labels}
    return lower_map.get(cleaned.lower(), cleaned)
