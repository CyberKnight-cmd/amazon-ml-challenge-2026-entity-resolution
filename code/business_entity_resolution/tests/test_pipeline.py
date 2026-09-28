import polars as pl

from entity_resolution.pipeline import assemble, write_submission


def test_assemble_one_row_per_entity_with_empty_lists():
    all_s1 = pl.DataFrame({"id": ["S1-3", "S1-1", "S1-2"]})
    acc = pl.DataFrame({"q_id": ["S3-9", "S2-5", "S2-7"], "s1_id": ["S1-1", "S1-1", "S1-3"]})
    out = assemble(all_s1, acc)
    assert out["source1_entity_id"].to_list() == ["S1-3", "S1-1", "S1-2"]  # order preserved
    assert out["matched_entity_ids"].to_list() == ["S2-7", "S2-5,S3-9", ""]


def test_written_file_has_truly_empty_fields_and_no_quotes(tmp_path):
    out = assemble(pl.DataFrame({"id": ["S1-1", "S1-2", "S1-3"]}), pl.DataFrame({"q_id": ["S2-5", "S3-9"], "s1_id": ["S1-1", "S1-1"]}))
    f = tmp_path / "matching_results.tsv"
    write_submission(out, f)
    assert f.read_bytes() == b"source1_entity_id\tmatched_entity_ids\nS1-1\tS2-5,S3-9\nS1-2\t\nS1-3\t\n"
