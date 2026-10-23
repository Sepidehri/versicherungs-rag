"""BM25 keyword search with German-aware tokenisation.

Keyword search complements dense retrieval: it reliably matches exact terms such as
"Marderbiss", "PIR" or "Wertschutzschrank" that embedding models sometimes blur.
"""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer
from rank_bm25 import BM25Okapi

GERMAN_STOPWORDS = frozenset(
    """
    aber alle allem allen aller alles als also am an ander andere anderem anderen anderer
    anderes auch auf aus bei bin bis bist da damit dann das dass dein deine dem den der des
    dessen deshalb die dies diese diesem diesen dieser dieses doch dort du durch ein eine
    einem einen einer eines er es etwa euer für gegen hab habe haben hat hatte hier hin ich
    ihr ihre im in ist ja jede jedem jeden jeder jedes jedoch kann kein keine keinem keinen
    man mein meine meinem meinen meiner mich mir mit muss nach nicht noch nun nur ob oder
    ohne sehr sein seine sich sie sind so soll sondern sowie um und uns unser unter vom von
    vor wann war was weil welche welchem welchen welcher wenn wer werden wie wieder will wir
    wird wo wurde zu zum zur über
    """.split()  # noqa: SIM905
)

_stemmer = snowballstemmer.stemmer("german")


@lru_cache(maxsize=50_000)
def _stem(word: str) -> str:
    return _stemmer.stemWord(word)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # Keep thousand-separated numbers together: "3.000" -> "3000".
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
    words = re.findall(r"[a-zäöüß0-9]+", text)
    return [_stem(w) for w in words if w not in GERMAN_STOPWORDS and len(w) > 1]


class BM25Index:
    def __init__(self, texts: list[str]) -> None:
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in ranked if scores[i] > 0]
