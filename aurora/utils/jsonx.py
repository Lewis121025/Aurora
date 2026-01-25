from __future__ import annotations

import json
from typing import Any

import orjson


def dumps(obj: Any) -> str:
    return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")


def loads(s: str) -> Any:
    return orjson.loads(s)
