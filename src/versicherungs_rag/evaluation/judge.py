"""Optional LLM-as-a-judge for faithfulness.

The string-based metrics check *whether* the right facts are in the answer. The judge
checks something they cannot: whether the answer claims anything that is not backed by
the retrieved excerpts (a hallucination), even if the expected facts are present.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from versicherungs_rag.llm.base import LLMClient, LLMError
from versicherungs_rag.schema import RAGAnswer

JUDGE_SYSTEM = """Du bewertest Antworten eines Frage-Antwort-Systems zu Versicherungsbedingungen.
Prüfe ausschließlich, ob jede Aussage der Antwort durch die Auszüge gedeckt ist.
Antworte nur mit JSON: {"faithfulness": <1-5>, "begruendung": "<ein Satz>"}
5 = jede Aussage ist belegt; 3 = teilweise belegt oder ungenau;
1 = wesentliche Aussagen widersprechen den Auszügen oder sind erfunden.
Eine Antwort, die korrekt sagt, dass die Auszüge keine Information enthalten, ist mit 5 zu bewerten."""


@dataclass
class JudgeVerdict:
    faithfulness: int | None
    reason: str


class FaithfulnessJudge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def judge(self, answer: RAGAnswer) -> JudgeVerdict:
        excerpts = "\n\n".join(f"[{rc.rank}] {rc.chunk.text}" for rc in answer.contexts)
        user = f"Auszüge:\n{excerpts}\n\nFrage: {answer.question}\n\nAntwort: {answer.answer}"
        try:
            text = self.llm.generate(JUDGE_SYSTEM, user, max_tokens=200).text
        except LLMError as exc:
            return JudgeVerdict(None, f"judge error: {exc}")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            data = json.loads(match.group(0)) if match else {}
            score = int(data["faithfulness"])
            return JudgeVerdict(score if 1 <= score <= 5 else None, str(data.get("begruendung", "")))
        except (ValueError, KeyError, TypeError):
            return JudgeVerdict(None, f"unparseable judge output: {text[:120]}")
