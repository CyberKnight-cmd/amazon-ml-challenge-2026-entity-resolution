import polars as pl
import pytest

from entity_resolution.metrics import macro_f05


def _long(rows):
    return pl.DataFrame(rows, schema={"s1": pl.String, "other": pl.String}, orient="row")


def test_perfect_prediction_is_one():
    truth = _long([("a", "x"), ("a", "y"), ("b", "z")])
    m = macro_f05(pl.DataFrame({"s1": ["a", "b", "c"]}), truth, truth)
    assert m["f05"] == pytest.approx(1.0)  # c is a correctly-empty singleton


def test_false_match_on_singleton_scores_zero():
    truth = _long([("a", "x")])
    pred = _long([("a", "x"), ("c", "q")])
    m = macro_f05(pl.DataFrame({"s1": ["a", "c"]}), truth, pred)
    assert m["f05"] == pytest.approx(0.5)
    assert m["singleton_f"] == 0.0


def test_missing_all_matches_scores_zero():
    truth = _long([("a", "x")])
    m = macro_f05(pl.DataFrame({"s1": ["a"]}), truth, _long([]))
    assert m["f05"] == 0.0


def test_f05_formula_favours_precision():
    truth = _long([("a", "1"), ("a", "2"), ("a", "3"), ("a", "4")])
    high_p = _long([("a", "1")])                       # P=1,   R=.25
    high_r = _long([("a", "1"), ("a", "2"), ("a", "3"), ("a", "4"), ("a", "9"), ("a", "8"), ("a", "7"), ("a", "6")])  # P=.5, R=1
    s = pl.DataFrame({"s1": ["a"]})
    fp, fr = macro_f05(s, truth, high_p)["f05"], macro_f05(s, truth, high_r)["f05"]
    assert fp == pytest.approx(1.25 * 1 * 0.25 / (0.25 + 0.25))
    assert fr == pytest.approx(1.25 * 0.5 * 1 / (0.125 + 1))
    assert fp > fr  # same amount of error, but the precision-heavy prediction scores higher


def test_precision_ignores_entities_with_empty_prediction():
    truth = _long([("a", "x"), ("b", "y")])
    pred = _long([("a", "x")])  # b: nothing predicted
    m = macro_f05(pl.DataFrame({"s1": ["a", "b"]}), truth, pred)
    assert m["matched_precision"] == 1.0 and m["predicted_share"] == 0.5 and m["matched_recall"] == 0.5
