"""Memory instrument control/report tests. No MLX or hardware substitution."""

import builtins
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import bonsai_memory_table as table


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "pack"
    root.mkdir()
    config = {"model_type": "prism_hadamard_qwen35", "text_config": {
        "num_hidden_layers": 64, "full_attention_interval": 4,
        "num_key_value_heads": 4, "head_dim": 256, "max_position_embeddings": 262144,
    }}
    (root / "config.json").write_text(json.dumps(config))
    (root / "mtplx_runtime.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"trunk and vision")
    (root / "mtp.safetensors").write_bytes(b"head")
    return root


def report():
    return table.make_report({"weights_bytes": 8_804_682_956,
                              "kv_bytes_per_token": 65536, "model_max_context": 262144},
                             [16, 18, 24], [4096, 8192, 16384])


def test_matrix_includes_the_tight_class_and_no_truncated_requests():
    rows = report()["rows"]
    assert len(rows) == 36
    assert {r["decode_tokens"] for r in rows} == {0, 1024}
    assert {r["kv_quantization"] for r in rows} == {"off", "q8"}
    tight = [r for r in rows if r["ram_gib"] == 16]
    assert len(tight) == 12
    # Before the 2026-09-21 measurement every 16 GiB row was refused; the
    # tight-machine rule now admits 8192 tokens there, so the 4K rows fit
    # the plan and the 8K-plus-decode and 16K rows are measured past it.
    assert all(r["planner_verdict"] == "admit" and r["planner"]["tight_machine"] for r in tight)
    assert all(r["planner"]["context_window_fit"] == 8192 for r in tight)
    assert [r["request_fits_planner"] for r in tight] == [
        r["total_tokens"] <= 8192 for r in tight
    ]
    assert max(r["total_tokens"] for r in tight) == 17408


def test_matrix_refuses_a_pack_the_margin_cannot_fund():
    rows = table.make_report({"weights_bytes": int(9.5 * table.GIB),
                              "kv_bytes_per_token": 65536, "model_max_context": 262144},
                             [16], [4096])["rows"]
    assert all(r["planner_verdict"] == "refuse" and not r["request_fits_planner"] for r in rows)


def test_dry_run_reads_real_file_sizes_and_cannot_import_mlx(pack, tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__

    def no_mlx(name, *args, **kwargs):
        assert not name.startswith(("mlx", "mtplx.runtime", "mtplx.server"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mlx)
    out = tmp_path / "dry"
    assert table.main(["--pack", str(pack), "--out", str(out), "--dry-run",
                       "--classes", "16", "18", "24", "--contexts", "4K", "8K", "16K"]) == 0
    assert not out.exists()
    assert "16384 | 1024 | q8" in capsys.readouterr().out
    metadata = table.pack_metadata(pack)
    assert metadata["weights_bytes"] == 20
    assert metadata["kv_bytes_per_token"] == 65536


def test_metal_limits_match_serve():
    calls = []
    mx = SimpleNamespace(set_memory_limit=lambda n: calls.append(("memory", n)),
                         set_wired_limit=lambda n: calls.append(("wired", n)))
    table.set_limits(mx, 18 * table.GIB, int(13.5 * table.GIB))
    assert calls == [("memory", 14_495_514_624), ("wired", 11_596_411_699)]


def test_peak_includes_loading_and_reports_a_completed_over_budget_case():
    row = report()["rows"][0]
    result = table.summarize_case(row, {"completed": True, "status": "completed",
                                      "request_peak_memory_bytes": 10 * table.GIB}, 13 * table.GIB)
    assert result["completed"]
    assert result["peak_memory_bytes"] == 13 * table.GIB
    assert result["peak_memory_gib"] == 13
    assert not result["within_engine_budget"]
    # The plan admits the 16 GiB class (tight machine); the measured peak,
    # not the verdict, is what says this case ran over the budget.
    assert result["planner_verdict"] == "admit"


def test_class_loads_once_and_runs_each_uncapped_case():
    rows = report()["rows"][:12]
    events, calls = [], []

    class Runner:
        def load(self, pack, row):
            calls.append("load")
            return {"load_peak_memory_bytes": 9 * table.GIB}

        def run_case(self, row):
            calls.append((row["prompt_tokens"], row["kv_quantization"], row["decode_tokens"]))
            return {"completed": True, "status": "completed", "request_peak_memory_bytes": 10 * table.GIB}

    table.measure_class(None, rows, events.append, Runner)
    assert calls.count("load") == 1
    assert len(calls) == 13
    assert calls[-1] == (16384, "q8", 1024)
    assert len(events) == 13


def test_allocation_failure_is_recorded_and_later_cases_are_not_claimed_complete():
    events = []

    class Runner:
        mx = SimpleNamespace(get_peak_memory=lambda: 13 * table.GIB)

        def load(self, pack, row):
            return {"load_peak_memory_bytes": 9 * table.GIB}

        def run_case(self, row):
            raise RuntimeError("[metal] Failed to allocate buffer")

    table.measure_class(None, report()["rows"][:12], events.append, Runner)
    first = events[1]["row"]
    assert first["allocation_failure"] and not first["completed"]
    assert first["peak_memory_gib"] == 13
    assert first["error_type"] == "RuntimeError"
    assert all(e["row"]["status"] == "not_run_after_failure" for e in events[2:])


def test_unavailable_metal_leaves_no_fake_peak():
    events = []

    def unavailable():
        raise ImportError("No Metal device available")

    table.measure_class(None, report()["rows"][:12], events.append, unavailable)
    assert len(events) == 13
    assert all(e["row"]["peak_memory_bytes"] is None and not e["row"]["completed"] for e in events[1:])
    assert not events[0]["allocation_failure"]


def test_report_has_both_peak_units_and_no_speed_fields(tmp_path):
    data = report()
    data["rows"][0] = table.summarize_case(data["rows"][0], {
        "completed": True, "status": "completed", "request_peak_memory_bytes": 10 * table.GIB,
    }, 9 * table.GIB)
    table.write_report(data, tmp_path)
    raw = (tmp_path / "memory.json").read_text()
    card = (tmp_path / "memory.md").read_text()
    assert "10737418240" in raw and "peak_memory_gib" in raw
    assert "10737418240" in card and "10.0000" in card
    assert not any(word in raw.lower() for word in ("tok_s", "tokens_per_second", "elapsed", "duration", "latency", "throughput"))


def test_prompt_is_exact_and_does_not_add_special_tokens():
    def encode(text, *, add_special_tokens):
        assert not add_special_tokens
        return [3, 5, 7]
    ids = table.exact_prompt(SimpleNamespace(encode=encode), 4096)
    assert len(ids) == 4096 and ids[-1] == 3


def test_pack_speed_metadata_is_not_copied_into_memory_report(pack):
    (pack / "mtplx_runtime.json").write_text(json.dumps({
        "public_model_id": "bonsai", "speed_evidence": {"tok_s": 123.45},
    }))
    metadata = table.pack_metadata(pack)
    assert metadata["identity"] == {"public_model_id": "bonsai"}
    assert "123.45" not in json.dumps(metadata)


@pytest.mark.parametrize("quant", ["off", "q8"])
def test_gpu_adapter_executes_full_decode_and_checks_actual_kv(quant, monkeypatch):
    # The real adapter runs in a dedicated child process. Keep its environment
    # private here too: the final q8 case must not change later cache tests.
    case_env = {}
    monkeypatch.setattr(table, "os", SimpleNamespace(environ=case_env))
    calls = []
    entry = SimpleNamespace(is_trimmable=lambda: True, kv_quant=quant == "q8",
                            kv_quant_config=SimpleNamespace(normalized_mode=quant))

    def prefill(rt, ids, *, return_hidden):
        assert len(ids) == 16384
        assert not return_hidden
        return [entry], object(), None, 999.0  # timing must never enter the report

    monkeypatch.setitem(sys.modules, "mtplx.generation", SimpleNamespace(
        _prefill=prefill, _eval_cache_roots=lambda cache: None,
    ))
    runner = object.__new__(table.MetalRunner)
    runner.mx = SimpleNamespace(
        clear_cache=lambda: None, reset_peak_memory=lambda: None, synchronize=lambda: None,
        argmax=lambda logits, axis: SimpleNamespace(reshape=lambda *shape: object()),
        eval=lambda *values: None, get_peak_memory=lambda: 10 * table.GIB,
        get_active_memory=lambda: 9 * table.GIB, get_cache_memory=lambda: 100,
    )
    runner.rt = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3]),
        forward_ar=lambda *args, **kwargs: calls.append(kwargs),
    )
    row = next(r for r in report()["rows"] if r["prompt_tokens"] == 16384
               and r["decode_tokens"] == 1024 and r["kv_quantization"] == quant)
    measured = runner.run_case(row)
    assert case_env == {
        "MTPLX_PAGED_KV_QUANT": quant,
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": quant,
        "MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS": "16384",
        "MTPLX_DYNAMIC_PAGED_KV_TOKENS": "17408",
    }
    assert measured["prefilled_tokens"] == 16384
    assert measured["decoded_tokens"] == len(calls) == 1024
    assert measured["observed_kv_quantization"] == quant
    assert "999" not in json.dumps(measured)


def test_q8_case_refuses_partial_or_silent_unquantized_caches():
    off = SimpleNamespace(is_trimmable=lambda: True, kv_quant=False)
    q8 = SimpleNamespace(is_trimmable=lambda: True, kv_quant=True,
                         kv_quant_config=SimpleNamespace(normalized_mode="q8"))
    with pytest.raises(RuntimeError, match="requested KV q8"):
        table.cache_quantization([q8, off], "q8")
