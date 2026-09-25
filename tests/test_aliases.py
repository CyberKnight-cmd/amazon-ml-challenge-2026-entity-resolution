import polars as pl

from entity_resolution.aliases import apply_aliases, fit_aliases


def _pairs():
    rows = [("acme private limited", "1 main street tx", "acme प्राइवेट लिमिटेड", "1 main st texas")] * 10
    return pl.DataFrame(rows, schema=["n1", "a1", "n2", "a2"], orient="row")


def test_fit_learns_aliases_by_position_and_apply_rewrites():
    tab = fit_aliases(_pairs(), min_total=5)
    m = dict(tab.filter(pl.col("field") == "n").select("q_tok", "s_tok").iter_rows())
    assert m["प्राइवेट"] == "private" and m["लिमिटेड"] == "limited"
    q = pl.DataFrame({"name_n": ["acme प्राइवेट लिमिटेड"], "addr_n": ["1 main st texas"], "is_domain": [False], "name_sq": ["x"]})
    r = apply_aliases(q, tab, mode="native")
    assert r["name_n"][0] == "acme private limited"
    assert r["addr_n"][0] == "1 main st texas"  # native mode leaves Latin address tokens alone
    assert apply_aliases(q, tab, mode="all")["addr_n"][0] == "1 main street tx"
