from __future__ import annotations


def levenshtein_distance(reference: str, prediction: str) -> int:
    """Compute Levenshtein edit distance using deterministic dynamic programming."""
    if reference == prediction:
        return 0
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)

    previous = list(range(len(prediction) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, pred_char in enumerate(prediction, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (ref_char != pred_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    """Compute CER as edit distance normalized by reference length."""
    if not reference:
        return 0.0 if not prediction else 1.0
    return levenshtein_distance(reference, prediction) / len(reference)
