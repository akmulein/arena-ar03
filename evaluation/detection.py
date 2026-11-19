"""Table localization metrics, independent of cell structure and OCR."""

from evaluation.matching import match_tables


def detection_summary(tp: int, fp: int, fn: int) -> dict:
    """Micro metrics; an empty prediction against empty GT scores one."""
    empty = float(tp + fp + fn == 0)
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    f1_denominator = 2 * tp + fp + fn
    return {
        'detection_precision': tp / precision_denominator if precision_denominator else empty,
        'detection_recall': tp / recall_denominator if recall_denominator else empty,
        'detection_f1': 2 * tp / f1_denominator if f1_denominator else empty,
        'detection_tp': tp,
        'detection_fp': fp,
        'detection_fn': fn,
    }


def score_detection(predicted: list[dict], truth: list[dict]) -> dict:
    """Score raw table detections independently of the recognition stages."""
    tp = len(match_tables(predicted, truth))
    return detection_summary(tp, len(predicted) - tp, len(truth) - tp)
