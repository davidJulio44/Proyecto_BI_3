from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime
from loguru import logger

HISTORY_PATH = Path("data/recommend_history.json")


def append_history(entry: Dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data: List[Dict] = []
    if HISTORY_PATH.exists():
        try:
            data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("History file exists but could not be parsed; starting new.")
    data.append(entry)
    HISTORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_history(limit: int = 50) -> List[Dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        return list(reversed(data))[:limit]
    except Exception as e:
        logger.error(f"Failed reading history: {e}")
        return []
