"""JSON schemas for the study's structured model outputs."""

import re


def object_schema(properties):
    """Require every named property and disallow additional JSON object keys."""
    return dict(
        type="object", properties=properties,
        required=list(properties), additionalProperties=False,
    )


def response_schema(req):
    """Build this stage's output schema; synthesis uses free text and returns None."""
    stage = req["stage"]
    text = {"type": "string"}
    if stage == "synthesis":
        return None
    if stage == "retrieval":
        return object_schema({
            "article_ids": {"type": "array", "items": text, "minItems": 10, "maxItems": 10},
        })
    if stage in {"annotation", "article_annotation"}:
        # Output keys must match the frame IDs presented in this particular prompt.
        frame_text = req["messages"][-1]["content"].split("\n\nFrames:\n", 1)[1].split("\n\n", 1)[0]
        frame_ids = re.findall(r"^\[([^\]]+)\]", frame_text, re.MULTILINE)
        return object_schema({
            frame_id: {"type": "integer", "enum": [0, 1, 2]} for frame_id in sorted(frame_ids)
        })
    if stage == "frame_extraction":
        point = object_schema({
            "info": text, "stance": {"type": "string", "enum": ["endorses", "refutes", "neutral"]},
        })
        return object_schema({
            "media_frame": text, "key_points": {"type": "array", "items": point},
        })
    if stage == "frame_grouping":
        group = object_schema({"name": text, "description": text})
        return object_schema({"groups": {"type": "array", "items": group}})
    if stage == "frame_assignment":
        user = req["messages"][-1]["content"]
        categories = user.split("\n\nCategories:\n", 1)[1].split("\n\nFrames:\n", 1)[0]
        names = [line[2:].split(": ", 1)[0] for line in categories.splitlines()]
        assignment = object_schema({
            "frame_id": {"type": "integer"},
            "group": {"type": "string", "enum": names},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        })
        return object_schema({"assignments": {"type": "array", "items": assignment}})
    raise ValueError(f"Unknown model stage: {stage}")
