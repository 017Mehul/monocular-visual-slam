import pytest

from scripts.evaluate_loop_closure import precision_recall


def test_precision_recall_reports_query_level_false_negatives():
    metrics = precision_recall(tp=2, fp=1, fn=2)
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(4 / 7)


def test_precision_recall_handles_no_predictions():
    metrics = precision_recall(tp=0, fp=0, fn=3)
    assert metrics["precision"] is None
    assert metrics["recall"] == 0.0
    assert metrics["f1"] is None
