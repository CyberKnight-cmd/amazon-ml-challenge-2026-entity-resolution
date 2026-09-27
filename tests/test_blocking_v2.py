import polars as pl

from entity_resolution.blocking_v2 import (
    V2Config,
    alias_classes,
    build_v2_index,
    canonical,
    leet,
    name_keys,
    union_candidates,
)

STOP = frozenset({"inc", "llc", "private", "limited"})


def _shared(a: str, b: str, max_del: int = 2) -> set[str]:
    return set(name_keys(a, STOP, max_del)) & set(name_keys(b, STOP, max_del))


def test_leet_repairs_words_but_not_numbers_or_ordinals():
    assert leet("5afe") == "safe" and leet("c0nsultancy") == "consultancy" and leet("8aker") == "baker"
    assert leet("4211") == "4211" and leet("27th") == "27th"


def test_name_bag_survives_reorder_added_dropped_and_typod_words():
    assert "nb:industries kiran limited private" in _shared("kiran industries private limited", "private kiran industries limited")
    assert _shared("urology health inc", "urology health inc trading")  # one added word
    assert _shared("martinez holding company llc", "martinez holdig company llc")  # one typo'd word
    assert _shared("city builders private limited", "smt city builders private ltd")  # prefix + suffix swap
    assert not _shared("acme inc", "zenith llc")


def test_anagram_key_links_glued_reordered_names():
    assert any(k.startswith("ag:") for k in _shared("lovato superior optics pllc", "opticssuperiorlovato"))


def test_alias_classes_pick_one_spelling_for_both_directions():
    table = pl.DataFrame({
        "field": ["n", "n", "a"], "q_tok": ["ltd", "limited", "texas"], "s_tok": ["limited", "ltd", "tx"],
        "cnt": [90, 90, 90], "total": [100, 100, 100], "purity": [0.9, 0.9, 0.9],
    })
    cls = alias_classes(table)
    df = pl.DataFrame({"name_n": ["acme limited", "acme ltd"], "addr_n": ["1 main texas", "1 main tx"]})
    out = canonical(df, cls)
    assert out["name_n"].to_list() == ["acme ltd", "acme ltd"] and out["addr_n"].to_list() == ["1 main tx", "1 main tx"]


def test_union_keeps_v1_pairs_and_adds_v2_pairs_with_evidence():
    s1 = pl.DataFrame({
        "id": ["S1-1", "S1-2", "S1-3"],
        "name_n": ["kiran industries private limited", "zenith tools inc", "acme bakery llc"],
        "addr_n": ["42 srirampore road kolkata", "4211 roosevelt street phoenix", "9 elm street austin"],
        "name_sq": ["kiranindustriesprivatelimited", "zenithtoolsinc", "acmebakeryllc"],
    })
    q = pl.DataFrame({
        "id": ["S2-1", "S3-2"],
        "name_n": ["private kiran industries limited", "zenith tools"],
        "addr_n": ["kolkata door no 279 42", "4210 roosevelt st phoenix"],
        "name_sq": ["privatekiranindustrieslimited", "zenithtools"],
    })
    idx = build_v2_index(s1, None, V2Config(top_k=5, stop_frac=0.9))
    v1 = pl.DataFrame({"q_id": ["S3-2"], "s1_id": ["S1-3"], "score": [1.0]})
    out = union_candidates(idx, q, v1)
    pairs = set(out.select("q_id", "s1_id").iter_rows())
    assert {("S2-1", "S1-1"), ("S3-2", "S1-2"), ("S3-2", "S1-3")} <= pairs
    kept_v1 = out.filter((pl.col("q_id") == "S3-2") & (pl.col("s1_id") == "S1-3"))
    assert kept_v1["in_v1"][0] == 1 and kept_v1["in_v2"][0] == 0
    found = out.filter((pl.col("q_id") == "S3-2") & (pl.col("s1_id") == "S1-2"))
    assert found["in_v2"][0] == 1 and found["score"][0] == 0 and found["v2_k_hn"][0] > 0  # 4210 ~ 4211
