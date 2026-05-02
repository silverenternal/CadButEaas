"""Room-label text normalization and lexicon features for floorplan spaces."""

from __future__ import annotations

import re
from collections.abc import Iterable


ROOM_TEXT_LABELS = [
    "balcony",
    "bathroom",
    "bedroom",
    "closet",
    "corridor",
    "kitchen",
    "living_room",
    "office",
    "storage",
    "toilet",
]


ROOM_TEXT_KEYWORDS = {
    "balcony": [
        "balcony",
        "terrace",
        "terassi",
        "kuisti",
        "parveke",
        "vilpola",
        "veranta",
        "patio",
        "lasikuisti",
        "avoterassi",
        "kattoterassi",
    ],
    "bathroom": ["bath", "bathroom", "shower", "ph", "kh", "pesuh", "pesuhuone", "pesu", "kph", "psh", "sh", "suihku", "sauna", "pe"],
    "bedroom": ["bed", "br", "bedroom", "mh", "makuuhuone"],
    "closet": ["closet", "wardrobe", "cl", "vh", "vaatehuone", "pukuh", "pukuhuone", "pkh", "puku"],
    "corridor": ["hall", "corridor", "entry", "entrance", "et", "tk", "aula", "eteinen", "kaytava", "käytävä", "halli", "yla aula", "ylä aula"],
    "kitchen": ["kit", "kitchen", "k", "keittiö", "keit", "kk", "tupak", "tupakeittiö", "apuk", "apukeittiö", "avok"],
    "living_room": ["living", "liv", "lounge", "oh", "olohuone", "rt", "r", "ruok", "ruokailu", "rh"],
    "office": ["office", "study", "työh", "tyohuone", "työhuone", "th", "kirjasto", "toimisto", "arkisto"],
    "storage": [
        "storage",
        "store",
        "utility",
        "laundry",
        "khh",
        "var",
        "tekn",
        "varasto",
        "kodinhoito",
        "autotalli",
        "at",
        "katt h",
        "oljy",
        "autovaja",
        "pannuh",
        "vaja",
        "ljh",
        "puuvar",
        "kuiv h",
        "aitta",
        "polttoaine",
        "laiteh",
        "kellari",
        "sail",
        "säil",
        "puuliiteri",
    ],
    "toilet": ["wc", "toilet", "w c"],
}


def normalize_room_text(value: str) -> str:
    text = value.lower().replace("ö", "o").replace("ä", "a").replace("å", "a")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def room_text_keyword_matches(label: str, text: str) -> bool:
    normalized = normalize_room_text(text)
    tokens = set(normalized.split())
    for keyword in ROOM_TEXT_KEYWORDS.get(label, []):
        normalized_keyword = normalize_room_text(keyword)
        if len(normalized_keyword) <= 3:
            if normalized_keyword in tokens:
                return True
        elif normalized_keyword in normalized:
            return True
    return False


def room_text_match_vector(texts: Iterable[str]) -> dict[str, float]:
    scores = {label: 0.0 for label in ROOM_TEXT_LABELS}
    for text in texts:
        for label in ROOM_TEXT_LABELS:
            if room_text_keyword_matches(label, text):
                scores[label] += 1.0
    return scores
