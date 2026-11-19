"""Table localization metrics, independent of cell structure and OCR."""

from evaluator.metrics import match_tables


def detection_summary(tp, fp, fn):
    """Micro metrics; an empty prediction against empty GT scores one."""
    empty = float(tp + fp + fn == 0)
    return dict(
        detection_precision=tp / (tp + fp) if tp + fp else empty,
        detection_recall=tp / (tp + fn) if tp + fn else empty,
        detection_f1=2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else empty,
        detection_tp=tp,
        detection_fp=fp,
        detection_fn=fn,
    )


def score_detection(predicted, truth):
    """Match bbox-only table lists one-to-one at IoU >= 0.5."""
    tp = len(match_tables(predicted, truth))
    return detection_summary(tp, len(predicted) - tp, len(truth) - tp)
