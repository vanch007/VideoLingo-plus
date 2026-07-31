from core.config_utils import load_key
from core.llm_provider import LLMProviderConfig, build_completion_args, create_chat_client, fix_base_url
from core.all_tts_functions.tts_registry import list_selectable_tts_methods, list_tts_methods
from core.providers.provider_governance import list_provider_records, verify_provider_records
from core.providers.contracts import TTSRequest, TTSResult
from core.providers.mlx_tts import DotsBackend, IndexTTS2Backend, MlxTTSRouter, looks_vietnamese
from core.providers.quality import content_similarity, reference_leak_score
from core.providers.benchmarking import build_benchmark_plan


def test_fix_base_url_adds_v1_for_openai_compatible_roots():
    assert fix_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
    assert fix_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"


def test_default_llm_is_requested_siliconflow_glm_model():
    assert load_key("llm.provider") == "openai_compatible"
    assert load_key("api.base_url") == "https://api.siliconflow.cn"
    assert load_key("api.model") == "zai-org/GLM-4.5-Air"


def test_completion_args_skip_json_format_when_provider_does_not_support_it():
    cfg = LLMProviderConfig(
        name="omlx",
        model="local",
        base_url="http://127.0.0.1:8000/v1",
        api_key="1234",
        supports_json_object=False,
    )
    args = build_completion_args(cfg, [{"role": "user", "content": "hi"}], response_json=True)
    assert "response_format" not in args


def test_chat_client_receives_provider_timeout():
    cfg = LLMProviderConfig(
        name="omlx",
        model="local",
        base_url="http://127.0.0.1:8000/v1",
        api_key="1234",
        timeout_seconds=12,
    )
    client = create_chat_client(cfg)
    assert client.timeout == 12


def test_siliconflow_indextts2_provider_is_removed():
    assert "sf_indextts2" not in list_tts_methods()
    assert "mlx_indextts2" in list_tts_methods()


def test_unavailable_tts_routes_are_deleted_and_mlx_clone_routes_are_selectable():
    selectable = list_selectable_tts_methods()
    registered = list_tts_methods()
    for removed in (
        "openai_tts", "elevenlabs_tts", "cosyvoice3_tts", "edge_tts",
        "voxcpm_tts", "gpt_sovits", "index_tts2", "piper_tts",
        "indonesian_tts", "custom_tts",
    ):
        assert removed not in registered
    for backend in (
        "mlx_indextts2", "mlx_higgs_audio", "mlx_dots_tts", "mlx_zonos2",
        "mlx_moss_tts",
    ):
        assert backend in selectable


def test_provider_governance_lists_mlx_clone_routes():
    records = list_provider_records()
    tts = {item["name"]: item for item in records["tts"]}
    assert "openai_tts" not in tts
    assert tts["mlx_indextts2"]["recommended"] is True
    assert tts["mlx_moss_tts"]["implemented"] is True
    assert tts["mlx_moss_tts"]["status"] == "experimental"


def test_provider_verify_reports_deprecated_routes(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    report = verify_provider_records()
    findings = {(item["provider"], item["message"]) for item in report["findings"]}
    assert report["ok"] is False
    assert any(provider == "gemini_3_pro" and "eligible" in message for provider, message in findings)


def test_mlx_router_routes_vietnamese_to_indextts2():
    req = TTSRequest(
        text="Xin chào, bạn có khỏe không?",
        output_path="output/audio/segs/test.wav",
        target_language="vi",
    )
    assert looks_vietnamese(req.text)
    assert MlxTTSRouter().select_backend(req) == "indextts2"


def test_indextts2_uses_native_duration_fit_for_subtitle_slots(monkeypatch, tmp_path):
    captured = {}

    def fake_run(self, cmd, request):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr(IndexTTS2Backend, "_run", fake_run)
    request = TTSRequest(
        text="Timed line",
        output_path=str(tmp_path / "line.wav"),
        ref_audio=str(tmp_path / "ref.wav"),
        target_duration=2.5,
    )
    IndexTTS2Backend({"root": str(tmp_path), "seed": 42}).synthesize(request)
    assert "--target-duration" in captured["cmd"]
    assert "--fit-duration" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--seed") + 1] == "42"


def test_indextts2_keeps_safe_duration_budget_but_does_not_fit_overrun(monkeypatch, tmp_path):
    captured = {}

    def fake_run(self, cmd, request):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr(IndexTTS2Backend, "_run", fake_run)
    request = TTSRequest(
        text="This sentence cannot fit without losing words.",
        output_path=str(tmp_path / "line.wav"),
        ref_audio=str(tmp_path / "ref.wav"),
        target_duration=1.0,
        task_row={"est_dur": 2.0},
    )
    IndexTTS2Backend({"root": str(tmp_path), "fit_duration": True}).synthesize(request)
    assert "--target-duration" in captured["cmd"]
    assert "--fit-duration" not in captured["cmd"]


def test_indextts2_content_repair_keeps_budget_without_native_fit(monkeypatch, tmp_path):
    captured = {}

    def fake_run(self, cmd, request):
        captured["cmd"] = cmd
        return None

    monkeypatch.setattr(IndexTTS2Backend, "_run", fake_run)
    request = TTSRequest(
        text="Generate every word.",
        output_path=str(tmp_path / "line.wav"),
        ref_audio=str(tmp_path / "ref.wav"),
        target_duration=3.0,
        task_row={"est_dur": 2.0, "disable_native_fit": True},
    )
    IndexTTS2Backend({"root": str(tmp_path), "fit_duration": True}).synthesize(request)
    assert "--target-duration" in captured["cmd"]
    assert "--fit-duration" not in captured["cmd"]


def test_indextts2_batch_keeps_model_resident_and_omits_unsafe_duration_cap(
    monkeypatch, tmp_path
):
    import csv
    from types import SimpleNamespace

    from pydub import AudioSegment

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        csv_path = cmd[cmd.index("--input") + 1]
        with open(csv_path, encoding="utf-8") as handle:
            captured["rows"] = list(csv.DictReader(handle))
        output_dir = tmp_path / "generated_proxy"
        requested_output_dir = cmd[cmd.index("--output-dir") + 1]
        output_dir = __import__("pathlib").Path(requested_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, 3):
            AudioSegment.silent(duration=100, frame_rate=16000).export(
                output_dir / f"{index:04d}_{index:04d}.wav", format="wav"
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("core.providers.mlx_tts.subprocess.run", fake_run)
    requests = [
        TTSRequest(
            text="Safe line",
            output_path=str(tmp_path / "safe.wav"),
            ref_audio=str(tmp_path / "ref.wav"),
            target_duration=2.0,
            task_row={"est_dur": 2.0},
        ),
        TTSRequest(
            text="Overlong line that must first be rewritten",
            output_path=str(tmp_path / "unsafe.wav"),
            ref_audio=str(tmp_path / "ref.wav"),
            target_duration=1.0,
            task_row={"est_dur": 2.0},
        ),
    ]

    results = IndexTTS2Backend(
        {"root": str(tmp_path), "fit_duration": True, "seed": 42}
    ).synthesize_batch(requests)

    assert captured["cmd"][1:3] == ["run", "mlx-indextts"]
    assert "batch" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--seed") + 1] == "42"
    assert captured["rows"][0]["target_duration_s"] == "2.000"
    assert captured["rows"][0]["fit_duration"] == "true"
    assert captured["rows"][1]["target_duration_s"] == "1.000"
    assert captured["rows"][1]["fit_duration"] == "false"
    assert len(results) == 2


def test_mlx_router_ignores_excel_nan_backend_values():
    req = TTSRequest(
        text="Xin chào",
        output_path="output/audio/segs/test.wav",
        target_language="vi",
        metadata={"backend": float("nan")},
    )
    assert MlxTTSRouter().select_backend(req) == "indextts2"


def test_mlx_router_routes_chinese_to_omnivoice_by_default():
    req = TTSRequest(
        text="你好，今天我们继续讲这个产品。",
        output_path="output/audio/segs/test.wav",
        target_language="zh",
    )
    assert MlxTTSRouter().select_backend(req) == "omnivoice"


def test_mlx_router_never_auto_selects_qwen_for_clean_reference():
    req = TTSRequest(
        text="Taxed ninety years ahead.",
        output_path="output/audio/segs/test.wav",
        target_language="en",
        ref_audio="output/audio/refers/1.wav",
        ref_text="前几任县长把税收到九十年以后了。",
    )
    assert MlxTTSRouter().select_backend(req) == "indextts2"


def test_mlx_router_accepts_skill_registry_aliases():
    req = TTSRequest(
        text="Xin chào",
        output_path="output/audio/segs/test.wav",
        target_language="vi",
        metadata={"backend": "indextts"},
    )
    assert MlxTTSRouter().select_backend(req) == "indextts2"


def test_mlx_router_accepts_all_new_voice_clone_aliases():
    aliases = {
        "mlx_higgs_audio": "higgs",
        "mlx_dots_tts": "dots",
        "mlx_zonos2": "zonos2",
        "mlx_moss_tts": "moss",
    }
    for alias, expected in aliases.items():
        req = TTSRequest(
            text="Hello",
            output_path="output/audio/segs/test.wav",
            metadata={"backend": alias},
        )
        assert MlxTTSRouter().select_backend(req) == expected


def test_dots_does_not_turn_target_duration_into_token_cap_by_default(monkeypatch, tmp_path):
    commands = []
    backend = DotsBackend({"root": str(tmp_path), "command_prefix": ["dots"]})
    monkeypatch.setattr(
        backend,
        "_run",
        lambda cmd, request: commands.append(cmd)
        or TTSResult(output_path=request.output_path, backend="dots"),
    )
    request = TTSRequest(
        text="Taxed ninety years ahead.",
        output_path=str(tmp_path / "out.wav"),
        ref_audio=str(tmp_path / "ref.wav"),
        target_language="en",
        target_duration=2.0,
    )

    backend.synthesize(request)

    assert "--target-duration" not in commands[0]
    token_index = commands[0].index("--max-audio-tokens") + 1
    assert commands[0][token_index] == "30"


def test_quality_text_scores_are_bounded():
    assert content_similarity("Xin chao", "Xin chao") == 1.0
    assert 0.0 <= reference_leak_score("hello reference", "hello target") <= 1.0


def test_reference_leak_ignores_shared_numbers_for_chinese_source():
    assert reference_leak_score("优惠 200 元，价格 329", "Giảm 200, giá 329") == 0.0
    assert reference_leak_score("优惠 200 元", "优惠 200") > 0


def test_content_similarity_normalizes_vietnamese_number_words():
    assert content_similarity("mười chín, mười tám, mười bảy", "19, 18, 17") == 1.0
    assert content_similarity("Mười hai, mười một, mười", "12, 11, 10") == 1.0


def test_benchmark_plan_contract_for_tts(tmp_path):
    dataset = tmp_path / "tts_lines.yaml"
    dataset.write_text(
        """
version: 1
items:
  - id: vi_short_sync_001
    language: vi
    text: Xin chao
    target_duration: 2.0
""".lstrip(),
        encoding="utf-8",
    )

    plan = build_benchmark_plan("tts", str(dataset), "mlx_indextts2,mlx_qwen3_tts")

    assert plan.status == "ready"
    assert plan.dataset_status == "valid"
    assert plan.dataset_exists is True
    assert plan.providers == ["mlx_indextts2", "mlx_qwen3_tts"]
    assert "duration_ratio" in plan.metrics


def test_benchmark_plan_marks_seed_schema_only_media_fixtures(tmp_path):
    dataset = tmp_path / "asr_samples.yaml"
    dataset.write_text(
        """
version: 1
status: seed_schema_only
items:
  - id: zh_reference_placeholder_001
    language: zh
    audio: null
    expected_transcript: 欢迎来到本期节目。
    expected_time_window:
      start: 0.0
      end: 3.0
""".lstrip(),
        encoding="utf-8",
    )

    plan = build_benchmark_plan("asr", str(dataset), "stable_ts_mlx")

    assert plan.status == "seed_schema_only"
    assert plan.dataset_status == "seed_schema_only"
    assert "audio is empty" in plan.validation_warnings[0]


def test_benchmark_plan_validates_llm_jsonl(tmp_path):
    dataset = tmp_path / "translation_zh_vi.jsonl"
    dataset.write_text(
        '{"id":"case-1","source_language":"zh","target_language":"vi","source_lines":["你好"],"expected_line_count":1}\n',
        encoding="utf-8",
    )

    plan = build_benchmark_plan("llm", str(dataset), "deepseek_v4_pro")

    assert plan.status == "ready"
    assert plan.dataset_status == "valid"
    assert plan.validation_errors == []
