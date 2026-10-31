"""Prompts and answer parsing. The prompt is fixed across all runs."""
from __future__ import annotations

import json
import re

_BN_NUM = "০১২৩৪৫৬৭৮৯"


def _bn(n: int) -> str:
    return "".join(_BN_NUM[int(d)] for d in str(n))


_FORMAT = (
    'JSON আকারে উত্তর দাও: {"answer": "<সংক্ষিপ্ত উত্তর>", '
    '"explanation": "<এক-দুই বাক্যে বাংলায় ব্যাখ্যা>"}'
)

_WITH_CONTEXT = (
    "তুমি একজন সহায়ক সহকারী। নিচের অনুচ্ছেদগুলোর তথ্য ব্যবহার করে প্রশ্নটির উত্তর দাও। "
    "শুধু অনুচ্ছেদে থাকা তথ্য ব্যবহার করবে। উত্তর অবশ্যই বাংলায় এবং যতটা সম্ভব সংক্ষিপ্ত "
    "(একটি শব্দ, বাক্যাংশ বা ছোট বাক্য) হবে। অনুচ্ছেদে উত্তর না থাকলে \"অনুচ্ছেদে উত্তর নেই\" লেখো।\n\n"
    "{contexts}\n\nপ্রশ্ন: {question}\n\n" + _FORMAT
)

_NO_CONTEXT = (
    "তুমি একজন সহায়ক সহকারী। নিজের জ্ঞান থেকে প্রশ্নটির উত্তর দাও। "
    "উত্তর অবশ্যই বাংলায় এবং যতটা সম্ভব সংক্ষিপ্ত (একটি শব্দ, বাক্যাংশ বা ছোট বাক্য) হবে।\n\n"
    "প্রশ্ন: {question}\n\n" + _FORMAT
)


def build_prompt(question: str, contexts: list[str] | None) -> str:
    if not contexts:
        return _NO_CONTEXT.replace("{question}", question)
    block = "\n".join(f"অনুচ্ছেদ [{_bn(i + 1)}]: {c}" for i, c in enumerate(contexts))
    return _WITH_CONTEXT.replace("{contexts}", block).replace("{question}", question)


def parse_response(text: str) -> tuple[str, str]:
    """Return (answer, explanation); falls back to the raw text if JSON is malformed."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj.get("answer", "")).strip(), str(obj.get("explanation", "")).strip()
    return text, ""
