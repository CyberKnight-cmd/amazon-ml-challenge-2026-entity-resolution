import polars as pl

from entity_resolution.decode import accept, top2


def test_top2_and_accept_gap_rule():
    scored = pl.DataFrame({"q_id": ["q1", "q1", "q2", "q3"], "s1_id": ["a", "b", "c", "d"], "p": [0.9, 0.85, 0.95, 0.4]})
    t = top2(scored).sort("q_id")
    assert t["s1_id"].to_list() == ["a", "c", "d"]
    assert t["p2"].to_list() == [0.85, 0.0, 0.0]
    # q1 has a near-tie -> rejected by the gap; q3 is below the threshold
    assert accept(t, 0.5, 0.1)["q_id"].to_list() == ["q2"]
    assert sorted(accept(t, 0.5, 0.0)["q_id"].to_list()) == ["q1", "q2"]
