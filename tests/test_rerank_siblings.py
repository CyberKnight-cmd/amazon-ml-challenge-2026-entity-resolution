import polars as pl

from entity_resolution.rerank import sibling_features


def test_sibling_features_reward_shared_per_source_variant_only():
    s1 = pl.DataFrame({"id": ["S1-1"], "name_n": ["lucky pub"], "addr_n": ["4211 roosevelt street phoenix az"]})
    n1, a1 = "lucky pub", "4211 roosevelt street phoenix az"
    b = pl.DataFrame({
        "q_id": ["S3-1", "S3-2", "S2-3"],
        "s1_id": ["S1-1"] * 3,
        "p": [0.95, 0.6, 0.5],
        "q_rank": [0, 0, 0],
        "src3": [1, 1, 0],
        "n1": [n1] * 3, "a1": [a1] * 3,
        "n2": ["lucky pub", "lucky pub", "lucky pub"],
        "a2": ["4210 roosevelt street capitol az", "4210 roosevelt st capitol arizona", "4211 roosevelt street glendale az"],
    })
    f = {r["q_id"]: r for r in sibling_features(b, s1).iter_rows(named=True)}
    assert f["S3-2"]["sib_addr_shared"] == 2 and f["S3-2"]["sib_addr_shared_same"] == 2  # shares 4210 + capitol
    assert f["S2-3"]["sib_addr_shared"] == 0 and f["S2-3"]["q_addr_rdiv"] == 1  # decoy: its own change only
    assert f["S3-1"]["sib_addr_shared"] == 0  # never its own sibling
    assert len(f) == 3
