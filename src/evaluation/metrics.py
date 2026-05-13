from __future__ import annotations

from dataclasses import dataclass

from src.data.schema import PagePrediction, RegionPrediction
from src.evaluation.cer import character_error_rate, levenshtein_distance
from src.ordering.reading_order import sort_regions_reading_order


TEXT_EVAL_EXCLUDED_TYPES = {"image", "graph"}


@dataclass(frozen=True)
class PageTextEvaluation:
    image_id: str
    reference_text: str
    predicted_text: str
    edit_distance: int
    reference_length: int
    page_cer: float


def region_is_text_evaluable(region: RegionPrediction) -> bool:
    if region.type in TEXT_EVAL_EXCLUDED_TYPES:
        return False
    if region.language is not None and region.language != "uk":
        return False
    if region.legibility is not None and region.legibility != "legible":
        return False
    return True


def assemble_page_text(regions: list[RegionPrediction] | tuple[RegionPrediction, ...]) -> str:
    ordered = sort_regions_reading_order(regions)
    texts = [region.text for region in ordered if region_is_text_evaluable(region) and region.text]
    return "\n".join(texts)


def evaluate_page_text(reference: PagePrediction, prediction: PagePrediction) -> PageTextEvaluation:
    reference_text = assemble_page_text(reference.regions)
    predicted_text = assemble_page_text(prediction.regions)
    edit_distance = levenshtein_distance(reference_text, predicted_text)
    page_cer = character_error_rate(reference_text, predicted_text)
    return PageTextEvaluation(
        image_id=reference.image_id,
        reference_text=reference_text,
        predicted_text=predicted_text,
        edit_distance=edit_distance,
        reference_length=len(reference_text),
        page_cer=page_cer,
    )


def mean_page_cer(evaluations: list[PageTextEvaluation]) -> float:
    if not evaluations:
        return 0.0
    total_edits = sum(item.edit_distance for item in evaluations)
    total_reference_length = sum(item.reference_length for item in evaluations)
    if total_reference_length == 0:
        return 0.0
    return total_edits / total_reference_length


def evaluate_pages(
    references: list[PagePrediction] | tuple[PagePrediction, ...],
    predictions: list[PagePrediction] | tuple[PagePrediction, ...],
) -> list[PageTextEvaluation]:
    predictions_by_id = {page.image_id: page for page in predictions}
    evaluations: list[PageTextEvaluation] = []
    for reference in references:
        prediction = predictions_by_id.get(reference.image_id)
        if prediction is None:
            raise KeyError(f"Missing prediction for image_id={reference.image_id}")
        evaluations.append(evaluate_page_text(reference, prediction))
    return evaluations
