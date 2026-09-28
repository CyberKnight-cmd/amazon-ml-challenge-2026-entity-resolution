import polars as pl

from entity_resolution.context import FEATURES, add_context


def _t2():
    # entity A owns three strong records (two S2, one S3) and one weak record; entity B has a single weak record
    return pl.DataFrame({
        "q_id": ["S2-1", "S2-2", "S3-3", "S2-4", "S3-5"],
        "s1_id": ["A", "A", "A", "A", "B"],
        "p1": [0.99, 0.95, 0.97, 0.40, 0.40],
        "s1_id2": [None, "B", None, None, "A"],
        "p2": [0.0, 0.30, 0.0, 0.0, 0.10],
    })


def test_context_counts_other_strong_records_excluding_self():
    t = add_context(_t2()).sort("q_id")
    weak_on_A = t.filter(pl.col("q_id") == "S2-4").row(0, named=True)
    weak_on_B = t.filter(pl.col("q_id") == "S3-5").row(0, named=True)
    assert weak_on_A["n_strong"] == 3 and weak_on_A["n_strong_s2"] == 2 and weak_on_A["n_strong_s3"] == 1
    assert weak_on_A["same_src_strong"] == 2 and abs(weak_on_A["max_other"] - 0.99) < 1e-9
    assert weak_on_B["n_strong"] == 0 and weak_on_B["max_other"] == 0.0
    strong = t.filter(pl.col("q_id") == "S2-1").row(0, named=True)
    assert strong["n_strong"] == 2  # the other two strong records, not itself
    assert abs(strong["max_other"] - 0.97) < 1e-9


def test_context_leak_is_probability_other_queries_put_on_runner_up_entity():
    t = add_context(_t2())
    assert abs(t.filter(pl.col("s1_id") == "B")["leak"][0] - 0.30) < 1e-9  # S2-2 has B as runner-up (p2=0.30)
    assert set(FEATURES) <= set(t.columns)
