"""Tests for the data catalog and the shipped data files."""

import pytest

from chiplens.catalog import (
    list_ids,
    load,
    spec_confidence,
    spec_value,
)


@pytest.mark.parametrize("kind", ["chips", "models", "assumptions"])
def test_every_shipped_data_file_loads_and_validates(kind: str) -> None:
    ids = list_ids(kind)
    assert ids, f"no files found for {kind}"
    for entity_id in ids:
        load(kind, entity_id)  # raises ValueError if the schema is broken


def test_expected_chips_and_models_present() -> None:
    assert {"sn40l", "h100_sxm", "h200_sxm", "gaudi3", "tpu_v5p", "tpu_v5e"} <= set(
        list_ids("chips")
    )
    assert "example_chip" in list_ids("chips")
    assert {"llama3_70b", "llama31_405b", "llama31_8b"} <= set(list_ids("models"))


def test_sn40l_published_numbers_match_paper_table() -> None:
    chip = load("chips", "sn40l")
    assert spec_value(chip, "peak_compute_bf16", "TFLOPS") == 638.0
    assert spec_value(chip, "hbm_capacity", "GB") == 64.0
    assert spec_value(chip, "hbm_bandwidth", "TB/s") == 1.8
    assert spec_value(chip, "ddr_capacity", "GB") == 1500.0
    assert spec_confidence(chip, "hbm_bandwidth") == "published"


def test_sn40l_power_is_flagged_assumed() -> None:
    assert spec_confidence(load("chips", "sn40l"), "board_power") == "assumed"


def test_unit_conversion() -> None:
    chip = load("chips", "sn40l")
    assert spec_value(chip, "ddr_bandwidth", "TB/s") == pytest.approx(0.2)
    assert spec_value(chip, "sram_capacity", "GB") == pytest.approx(0.52)


def test_wrong_unit_request_raises() -> None:
    with pytest.raises(ValueError, match="unit"):
        spec_value(load("chips", "sn40l"), "board_power", "TB/s")


def test_missing_spec_lists_available_keys() -> None:
    with pytest.raises(KeyError, match="board_power"):
        spec_value(load("chips", "sn40l"), "does_not_exist")


def test_unknown_entity_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load("chips", "no_such_chip")


def test_llama_param_count_matches_architecture_formula() -> None:
    from chiplens.llm import params_from_architecture

    m = load("models", "llama3_70b")
    derived = params_from_architecture(
        layers=int(spec_value(m, "layers")),
        hidden_size=int(spec_value(m, "hidden_size")),
        heads=int(spec_value(m, "attention_heads")),
        kv_heads=int(spec_value(m, "kv_heads")),
        ffn_size=int(spec_value(m, "ffn_size")),
        vocab_size=int(spec_value(m, "vocab_size")),
    )
    assert spec_value(m, "params_billion") == pytest.approx(derived, abs=1e-6)
