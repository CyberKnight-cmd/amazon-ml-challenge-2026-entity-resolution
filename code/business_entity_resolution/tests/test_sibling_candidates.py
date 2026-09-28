import polars as pl

from entity_resolution.rerank import add_sibling_candidates, sibling_candidates


def test_same_source_sibling_with_the_address_variant_proposes_its_entity():
    scored = pl.DataFrame({
        "q_id": ["S3-1", "S3-2", "S2-5"],
        "s1_id": ["S1-1", "S1-9", "S1-7"],
        "p": [0.95, 0.10, 0.97],
    })
    queries = pl.DataFrame({
        "id": ["S3-1", "S3-2", "S2-5"],
        "addr_n": ["1411 southwell dr candall city tx", "1411 souohwell dr candall city tx", "1411 souohwell dr candall city tx"],
    })
    sc = sibling_candidates(scored, queries)
    got = {(r["q_id"], r["s1_id"]): r["sc_n"] for r in sc.iter_rows(named=True)}
    assert got.get(("S3-2", "S1-1")) == 1  # found through its S3 sibling's shared variant '1411 ... candall'
    assert ("S3-2", "S1-7") not in got  # an S2 record never vouches for an S3 record
    out = add_sibling_candidates(scored, sc)
    new = out.filter(pl.col("sc_new") == 1)
    assert new.select("q_id", "s1_id").rows() == [("S3-2", "S1-1")] and new["p"][0] == 0.0
    assert out.height == 4 and out["sc_n"].null_count() == 0
