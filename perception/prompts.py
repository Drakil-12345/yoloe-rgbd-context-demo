"""Deterministic STT text -> ontology concept -> optimized YOLOE prompt."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .common import LIBRARY_PATH


def normalize_text(text: str) -> str:
    text = text.lower().replace("đ", "d")
    text = "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )
    return " ".join(re.findall(r"[a-z0-9]+", text))


def contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def best_window_similarity(text: str, alias: str) -> float:
    alias_tokens = alias.split()
    text_tokens = text.split()
    if not alias_tokens or not text_tokens:
        return 0.0
    scores = []
    expected = len(alias_tokens)
    for size in range(max(1, expected - 1), min(len(text_tokens), expected + 1) + 1):
        for start in range(len(text_tokens) - size + 1):
            window = " ".join(text_tokens[start : start + size])
            scores.append(difflib.SequenceMatcher(None, alias, window).ratio())
    return max(scores, default=0.0)


@dataclass
class ResolvedPrompt:
    original_text: str
    normalized_text: str
    concept: str | None
    display_name: str | None
    ontology_path: list[str]
    matched_alias: str | None
    match_type: str | None
    match_score: float
    optimized_prompt: str | None
    fallback_prompts: list[str]
    color: str | None
    spatial: list[str]
    selection: str | None
    excluded_concepts: list[str]
    ambiguous_with: list[str]
    prompt_policy: str


class PromptOptimizer:
    def __init__(self, library_path: Path | str):
        self.library_path = Path(library_path)
        self.library: dict[str, Any] = json.loads(self.library_path.read_text(encoding="utf-8"))
        self.alias_rows = []
        for concept, config in self.library["concepts"].items():
            for alias in config["aliases"]:
                self.alias_rows.append(
                    {
                        "concept": concept,
                        "original": alias["text"],
                        "normalized": normalize_text(alias["text"]),
                        "weight": float(alias.get("weight", 1.0)),
                    }
                )
        self.modifier_rows = {
            group: {
                canonical: sorted((normalize_text(alias) for alias in aliases), key=len, reverse=True)
                for canonical, aliases in values.items()
            }
            for group, values in self.library["modifiers"].items()
        }

    def _extract_modifier(self, text: str, group: str) -> list[str]:
        matches = []
        for canonical, aliases in self.modifier_rows[group].items():
            matching = [alias for alias in aliases if contains_phrase(text, alias)]
            if matching:
                specificity = max((len(alias.split()), len(alias)) for alias in matching)
                matches.append((specificity, canonical))
        # Specific phrases such as "xanh la" override the generic local
        # convention that bare "xanh" means blue.
        matches.sort(reverse=True)
        return [canonical for _, canonical in matches]

    @staticmethod
    def _split_clauses(user_text: str) -> list[tuple[str, bool]]:
        folded = user_text.lower().replace("đ", "d")
        folded = "".join(
            character
            for character in unicodedata.normalize("NFD", folded)
            if unicodedata.category(character) != "Mn"
        )
        folded = re.sub(r"\b(?:nhung|but|instead)\b", "|", folded)
        folded = re.sub(r"[,;.!?]+", "|", folded)
        clauses = []
        negative_markers = ("dung tim", "khong phai", "bo qua", "loai bo", "do not", "dont", "not", "ignore")
        for raw_clause in folded.split("|"):
            clause = normalize_text(raw_clause)
            if clause:
                clauses.append((clause, any(contains_phrase(clause, marker) for marker in negative_markers)))
        return clauses or [(normalize_text(user_text), False)]

    def _object_candidates(self, text: str):
        exact_candidates = []
        for row in self.alias_rows:
            alias = row["normalized"]
            if contains_phrase(text, alias):
                # Weight matters more for one-word generic aliases such as "chai".
                score = 0.9 + 0.04 * min(len(alias.split()), 4) + 0.1 * row["weight"]
                exact_candidates.append((score, len(alias), row, "exact_phrase", text))

        allowed_fuzzy_concepts = (
            {item[2]["concept"] for item in exact_candidates} if exact_candidates else None
        )
        fuzzy_candidates = []
        for row in self.alias_rows:
            alias = row["normalized"]
            if contains_phrase(text, alias) or len(alias) < 5:
                continue
            if allowed_fuzzy_concepts is not None and row["concept"] not in allowed_fuzzy_concepts:
                continue
            if len(alias) >= 5:
                similarity = best_window_similarity(text, alias)
                if similarity < 0.78:
                    continue
                # A likely two-word ASR correction can outrank an exact but generic
                # one-word alias ("chai nuok" should prefer "chai nuoc" over "chai").
                score = similarity * row["weight"] + 0.1 * min(len(alias.split()), 3)
                fuzzy_candidates.append((score, len(alias), row, "fuzzy_asr", text))
        return exact_candidates + fuzzy_candidates

    def resolve(self, user_text: str) -> ResolvedPrompt:
        text = normalize_text(user_text)
        candidates = []
        excluded = set()
        for clause, is_negated in self._split_clauses(user_text):
            clause_candidates = self._object_candidates(clause)
            if is_negated:
                excluded.update(item[2]["concept"] for item in clause_candidates)
            else:
                candidates.extend(clause_candidates)

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        if not candidates:
            colors = self._extract_modifier(text, "colors")
            spatial = self._extract_modifier(text, "spatial")
            selections = self._extract_modifier(text, "selection")
            return ResolvedPrompt(
                original_text=user_text,
                normalized_text=text,
                concept=None,
                display_name=None,
                ontology_path=[],
                matched_alias=None,
                match_type=None,
                match_score=0.0,
                optimized_prompt=None,
                fallback_prompts=[],
                color=colors[0] if colors else None,
                spatial=spatial,
                selection=selections[0] if selections else None,
                excluded_concepts=sorted(excluded),
                ambiguous_with=[],
                prompt_policy="No known object concept found.",
            )

        best_score, _, best_row, best_match_type, best_clause = candidates[0]
        concept = best_row["concept"]
        config = self.library["concepts"][concept]
        colors = self._extract_modifier(best_clause, "colors")
        spatial = self._extract_modifier(best_clause, "spatial")
        selections = self._extract_modifier(best_clause, "selection")
        prompt_rows = sorted(config["prompts"], key=lambda row: row.get("priority", 0), reverse=True)
        prompts = [row["text"] for row in prompt_rows]
        ambiguity = sorted(
            {
                row["concept"]
                for score, _, row, _, _ in candidates[1:]
                if row["concept"] != concept and best_score - score <= 0.05
            }
        )
        return ResolvedPrompt(
            original_text=user_text,
            normalized_text=text,
            concept=concept,
            display_name=config["display_name"],
            ontology_path=config["ontology_path"],
            matched_alias=best_row["original"],
            match_type=best_match_type,
            match_score=round(best_score, 4),
            optimized_prompt=prompts[0],
            fallback_prompts=prompts[1:],
            color=colors[0] if colors else None,
            spatial=spatial,
            selection=selections[0] if selections else None,
            excluded_concepts=sorted(excluded),
            ambiguous_with=ambiguity,
            prompt_policy=(
                "Use the empirically preferred base-class prompt for recall; "
                "keep color and spatial language as post-detection constraints."
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="STT keyword or utterance")
    parser.add_argument("--library", type=Path, default=LIBRARY_PATH)
    args = parser.parse_args()
    resolved = PromptOptimizer(args.library).resolve(" ".join(args.text))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(asdict(resolved), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
