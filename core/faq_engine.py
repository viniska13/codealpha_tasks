"""Retrieval-based FAQ engine for Task 2.

Pipeline
--------
1. **Preprocess**: lower-case, expand contractions, tokenise, drop stop-words,
   stem with a light suffix-stripping stemmer.
2. **Vectorise** every FAQ question + paraphrase with two TF-IDF models:
   * word 1-2 grams on the stemmed text   -> understands meaning / word order
   * character 3-5 grams on the raw text  -> tolerates typos ("certifcate")
3. **Score** with cosine similarity and blend the two (hybrid score).
4. **Decide**: above ``threshold`` -> answer; between ``floor`` and ``threshold``
   -> "did you mean ...?" suggestions; below -> out-of-scope fallback.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "faqs.json"

# Keep words that carry meaning in this domain even though they are generic stop-words
_KEEP = {"how", "many", "when", "where", "not", "no", "only", "one", "two", "three", "four"}
# generic request verbs add no topical signal ("explain X", "tell me about X")
_GENERIC = {"explain", "tell", "please", "pls", "want", "know", "need", "give", "show", "like"}
STOP_WORDS = frozenset(w for w in ENGLISH_STOP_WORDS if w not in _KEEP) | _GENERIC

_CONTRACTIONS = {
    "what's": "what is", "it's": "it is", "i'm": "i am", "don't": "do not",
    "doesn't": "does not", "can't": "can not", "won't": "will not",
    "isn't": "is not", "i've": "i have", "i'll": "i will", "that's": "that is",
    "where's": "where is", "how's": "how is", "who's": "who is",
}
_TOKEN = re.compile(r"[a-z0-9]+")

_SUFFIXES = ("ations", "ation", "ingly", "ness", "ments", "ment", "ings",
             "ing", "edly", "ies", "ied", "ers", "er", "ed", "es", "ly", "s")
STEMMER = "suffix-stripping stemmer"


def _stem(word: str) -> str:
    """Light suffix-stripping stemmer ("deadlines" -> "deadline", "submitting" -> "submitt")."""
    if len(word) <= 3 or word.isdigit():
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: -len(suf)]
            return stem + "y" if suf in ("ies", "ied") else stem
    return word


def preprocess(text: str) -> str:
    """Normalise text for the word-level model."""
    text = text.lower()
    for short, full in _CONTRACTIONS.items():
        text = text.replace(short, full)
    all_tokens = _TOKEN.findall(text)
    tokens = [t for t in all_tokens if t not in STOP_WORDS]
    # "what can you do" is made only of stop-words; keep them rather than return ""
    return " ".join(_stem(t) for t in (tokens or all_tokens))


@dataclass
class Match:
    faq_id: str
    category: str
    question: str
    answer: str
    score: float          # final hybrid score after the vocabulary-coverage penalty
    word: float = 0.0     # word-level TF-IDF cosine against the best-matching phrasing
    char: float = 0.0     # character n-gram TF-IDF cosine against the same phrasing
    phrase: str = ""      # the stored phrasing that matched best


@dataclass
class Analysis:
    """Everything the engine computed for one query (shown in the UI inspector)."""
    query: str
    tokens: list[str]     # preprocessed (stop-words removed, stemmed)
    coverage: float       # share of query words known to the FAQ vocabulary
    matches: list[Match]


@dataclass
class Response:
    kind: str  # "answer" | "suggest" | "fallback"
    answer: str
    best: Optional[Match]
    suggestions: list[Match]
    analysis: Optional[Analysis] = None

    @property
    def confidence(self) -> float:
        return self.best.score if self.best else 0.0


class FAQEngine:
    def __init__(self, faqs: Optional[list[dict]] = None, *, threshold: float = 0.38,
                 floor: float = 0.15, word_weight: float = 0.55):
        if faqs is None:
            faqs = json.loads(DEFAULT_DATA.read_text(encoding="utf-8"))
        if not faqs:
            raise ValueError("FAQ list is empty")
        self.faqs = faqs
        self.threshold, self.floor, self.word_weight = threshold, floor, word_weight
        self._by_id = {f["id"]: f for f in faqs}

        # one indexed document per question / paraphrase
        self._docs: list[str] = []
        self._owner: list[int] = []  # doc index -> faq index
        for i, f in enumerate(faqs):
            for q in [f["question"], *f.get("variants", [])]:
                self._docs.append(q)
                self._owner.append(i)
        self._owner_arr = np.array(self._owner)

        self._word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._word_mat = self._word_vec.fit_transform([preprocess(d) for d in self._docs])
        self._unigrams = [w for w in self._word_vec.vocabulary_ if " " not in w]
        self._char_mat = self._char_vec.fit_transform([d.lower() for d in self._docs])

    COVERAGE_FLOOR = 0.25

    def _coverage(self, processed_query: str) -> float:
        """Fraction of the query's content stems that exist in the FAQ vocabulary."""
        toks = processed_query.split()
        if not toks:
            return 0.0
        vocab = self._word_vec.vocabulary_
        known = 0
        for t in toks:
            # a near-miss spelling ("deadlne") still counts as a known word
            if t in vocab or difflib.get_close_matches(t, self._unigrams, n=1, cutoff=0.8):
                known += 1
        return known / len(toks)

    # ------------------------------------------------------------------ #
    def analyze(self, query: str, k: int = 3) -> Analysis:
        """Top-k distinct FAQs for ``query`` plus the intermediate values behind them."""
        query = query.strip()
        pq = preprocess(query) if query else ""
        if not query:
            return Analysis(query, [], 0.0, [])
        word_sim = (cosine_similarity(self._word_vec.transform([pq]), self._word_mat)[0]
                    if pq else np.zeros(len(self._docs)))
        char_sim = cosine_similarity(self._char_vec.transform([query.lower()]), self._char_mat)[0]
        coverage = self._coverage(pq)
        # TF-IDF silently drops out-of-vocabulary words, so "explain quantum computing"
        # would collapse to "explain". Penalise queries whose content words are unknown.
        penalty = self.COVERAGE_FLOOR + (1 - self.COVERAGE_FLOOR) * coverage
        sim = (self.word_weight * word_sim + (1 - self.word_weight) * char_sim) * penalty

        best_doc = np.zeros(len(self.faqs), dtype=int)
        best_score = np.zeros(len(self.faqs))
        for faq_idx in range(len(self.faqs)):
            docs = np.where(self._owner_arr == faq_idx)[0]
            d = docs[np.argmax(sim[docs])]
            best_doc[faq_idx], best_score[faq_idx] = d, sim[d]

        matches = []
        for i in np.argsort(-best_score)[:k]:
            f, d = self.faqs[i], best_doc[i]
            matches.append(Match(f["id"], f["category"], f["question"], f["answer"], float(best_score[i]),
                                 float(word_sim[d]), float(char_sim[d]), self._docs[d]))
        return Analysis(query, pq.split(), coverage, matches)

    def rank(self, query: str, k: int = 3) -> list[Match]:
        return self.analyze(query, k).matches

    def ask(self, query: str, k: int = 3, threshold: Optional[float] = None) -> Response:
        threshold = self.threshold if threshold is None else threshold
        analysis = self.analyze(query, k)
        ranked = analysis.matches
        if not ranked or ranked[0].score < self.floor:
            return Response(
                "fallback",
                "I couldn't find that in my FAQ set. I can help with the internship's **dates, "
                "deadline, tasks, GitHub/LinkedIn steps, submission form, certificate, LOR and "
                "support contacts**. For anything else, message CodeAlpha on WhatsApp.",
                ranked[0] if ranked else None, ranked, analysis)
        best = ranked[0]
        if best.score >= threshold:
            return Response("answer", best.answer, best, ranked[1:], analysis)
        return Response("suggest", "I'm not fully sure what you meant. Did you mean one of these?",
                        best, ranked, analysis)

    def categories(self) -> list[str]:
        seen: list[str] = []
        for f in self.faqs:
            if f["category"] not in seen:
                seen.append(f["category"])
        return seen

    def questions_in(self, category: str) -> list[dict]:
        return [f for f in self.faqs if f["category"] == category]

    def get(self, faq_id: str) -> dict:
        return self._by_id[faq_id]


def evaluate(engine: "FAQEngine", data: dict) -> dict:
    """Score the engine on ``{"in_scope": [[query, faq_id], ...], "out_of_scope": [query, ...]}``."""
    top1 = top3 = confident = 0
    misses = []
    for query, expected in data["in_scope"]:
        ranked = engine.rank(query, 3)
        ids = [m.faq_id for m in ranked]
        top1 += ids[0] == expected
        top3 += expected in ids
        resp = engine.ask(query)
        confident += resp.kind == "answer" and resp.best.faq_id == expected
        if ids[0] != expected:
            misses.append({"query": query, "expected": expected, "got": ids[0],
                           "score": round(ranked[0].score, 2)})
    leaks = []
    for query in data["out_of_scope"]:
        resp = engine.ask(query)
        if resp.kind == "answer":
            leaks.append({"query": query, "answered_with": resp.best.faq_id, "score": round(resp.confidence, 2)})
    n, m = len(data["in_scope"]), len(data["out_of_scope"])
    return {"n_in_scope": n, "n_out_of_scope": m, "top1": top1 / n, "top3": top3 / n,
            "confident_correct": confident / n, "out_of_scope_rejected": (m - len(leaks)) / m,
            "misses": misses, "leaks": leaks}
