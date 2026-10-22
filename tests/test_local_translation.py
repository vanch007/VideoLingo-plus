from dataclasses import replace
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from core import ask_gpt, llm_provider, local_translation, translation_review, translation_state
from core.translation_language import translation_language_name


def test_preset_environment_overrides_are_effective(monkeypatch):
    monkeypatch.setenv("OMLX_MODEL", "index-test-model")
    monkeypatch.setenv("OMLX_BASE_URL", "http://localhost:8900/v1")
    monkeypatch.setenv("OMLX_API_KEY", "test-only")
    provider = llm_provider.get_llm_provider_config()
    assert (provider.model, provider.base_url, provider.api_key) == (
        "index-test-model", "http://localhost:8900/v1", "test-only")


def test_cache_fingerprint_tracks_sampling_and_endpoint_not_secret():
    config = llm_provider.get_llm_provider_config()
    fingerprint = llm_provider.completion_fingerprint(config, True)
    assert fingerprint != llm_provider.completion_fingerprint(replace(config, temperature=0.7), True)
    assert fingerprint != llm_provider.completion_fingerprint(replace(config, base_url="http://localhost:8900/v1"), True)
    assert fingerprint != llm_provider.completion_fingerprint(config, False)
    assert fingerprint == llm_provider.completion_fingerprint(replace(config, api_key="different-secret"), True)


def test_truncated_but_repairable_json_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(ask_gpt, "LOG_FOLDER", str(tmp_path))
    choice = SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"issues": []'))
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[choice])
    completions = SimpleNamespace(create=create)
    monkeypatch.setattr(ask_gpt, "create_chat_client", lambda _: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    with pytest.raises(Exception, match="Incomplete LLM response"):
        ask_gpt.ask_gpt("review", log_title="truncated", max_retries=3)
    assert len(calls) == 1
    assert not (tmp_path / "truncated.json").exists()


def test_cache_does_not_reuse_another_endpoint_or_legacy_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(ask_gpt, "LOG_FOLDER", str(tmp_path))
    (tmp_path / "translate.json").write_text(json.dumps([
        {"model": "m", "prompt": "p", "response": {"text": "old"}},
        {"model": "m", "prompt": "p", "response": {"text": "new"}, "request_fingerprint": "new"},
    ]))
    assert ask_gpt.check_ask_gpt_history("p", "m", "translate", "missing") is False
    assert ask_gpt.check_ask_gpt_history("p", "m", "translate", "new") == {"text": "new"}


@pytest.mark.parametrize("response", [
    {"1": {"origin": "甲", "translation": "A"}},
    {"1": {"origin": "乙", "translation": "B"}, "2": {"origin": "甲", "translation": "A"}},
    {"1": {"origin": "甲", "translation": ""}, "2": {"origin": "乙", "translation": "B"}},
    {"1": {"origin": "甲", "translation": "[L001] A"}, "2": {"origin": "乙", "translation": "B"}},
    {"1": {"origin": "甲", "translation": "甲"}, "2": {"origin": "乙", "translation": "B"}},
    ["A", "B"],
])
def test_localized_translation_rejects_lost_identity_or_metadata(response):
    assert local_translation.validate_translation_rows(response, ["甲", "乙"], "translation", "en")["status"] == "error"


def test_json_member_order_does_not_change_subtitle_order(monkeypatch):
    response = {"2": "B", "1": "A"}
    calls = []
    def fake_ask(prompt, **kwargs):
        assert kwargs["valid_def"](response)["status"] == "success"
        calls.append(kwargs["log_title"])
        return response
    monkeypatch.setattr(local_translation, "ask_gpt", fake_ask)
    monkeypatch.setattr(local_translation, "effective_target_language", lambda _: "en")
    assert local_translation.translate_localized("甲\n乙", "speaker context") == ("A\nB", "甲\n乙")
    assert calls == ["translate_localized"]


@pytest.mark.parametrize("response", [{"1": "A"}, {"1": 3, "2": "B"}, {"1": "[S01] A", "2": "B"}, {"1": "", "2": "B"}])
def test_flat_translation_rejects_missing_rows_or_unusable_text(response):
    assert local_translation.validate_translation_rows(response, ["甲", "乙"], None, "en")["status"] == "error"


def test_copied_word_detection_preserves_names_acronyms_placeholders_and_glossary():
    source = "Use iPhone and OpenAI API with {order_id}, %s and https://example.com; enable pytorch for the audition."
    target = "用iPhone和OpenAI API，保留{order_id}、%s及https://example.com；启用pytorch，明天audition加油。"
    assert local_translation.source_copy_candidates(source, target, "zh", [{"src": "pytorch", "tgt": "pytorch"}]) == ["audition"]
    assert local_translation.source_copy_candidates(source, source, "vi") == []


def test_compound_names_are_not_split_into_false_untranslated_words():
    assert local_translation.source_copy_candidates(
        "Turn it into what we call B-roll.", "把它变成我们所说的B-roll。", "zh") == []
    # The label must not license ordinary vocabulary, including compounds.
    assert local_translation.source_copy_candidates(
        "B-roll is a well-known term; roll the footage.", "B-roll是well-known术语；roll素材。", "zh") == ['roll', 'well-known']


def test_glossary_licenses_the_complete_name_without_leaking_word_exemptions():
    assert local_translation.source_copy_candidates(
        "Claude Code writes the code.", "Claude Code编写code。", "zh",
        [{'src': 'Claude Code', 'tgt': 'Claude Code'}]) == ['code']


def test_generated_code_formatting_cannot_hide_an_untranslated_word():
    assert local_translation.source_copy_candidates('Your audition is tomorrow.', '明天`audition`。', 'zh') == ['audition']
    assert local_translation.source_copy_candidates('Use `audition` as the key.', '把`audition`用作键。', 'zh') == []


def test_source_copy_repair_runs_only_for_suspect_rows(monkeypatch):
    calls = []
    def fake_ask(prompt, **kwargs):
        calls.append(kwargs["log_title"])
        if kwargs['log_title'] == 'translate_source_copy_terms':
            assert kwargs['response_json'] is True
            assert kwargs['valid_def']({'audition': 'audition'})['status'] == 'error'
            assert kwargs['valid_def']({'audition': '试镜'})['status'] == 'success'
            return {'audition': '试镜'}
        assert kwargs["response_json"] is False
        assert 'audition' in prompt and 'rejected_draft' in prompt
        assert '试镜' in prompt
        retry = kwargs['validation_retry_prompt'](prompt, '明天audition加油。', 'untranslated word')
        assert 'untranslated word' in retry and '【源文】' in retry
        assert kwargs["valid_def"]("明天试镜加油。")["status"] == "success"
        assert kwargs["valid_def"]("明天audition加油。")["status"] == "error"
        return "明天试镜加油。"
    monkeypatch.setattr(local_translation, "ask_gpt", fake_ask)
    assert local_translation.repair_source_word_copies("Your audition is tomorrow.", "明天audition加油。", "zh") == "明天试镜加油。"
    assert local_translation.repair_source_word_copies("Your audition is tomorrow.", "明天试镜加油。", "zh") == "明天试镜加油。"
    assert calls == ['translate_source_copy_terms', 'translate_source_copy_repair']


def test_other_models_keep_legacy_translation():
    config = replace(llm_provider.get_llm_provider_config(), model="general-model")
    assert llm_provider.get_translation_strategy(config) == "legacy"


def test_source_facts_override_inferred_role_only_on_affected_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    replies = [
        {"1": "I'll pay 2,000 to fix your car.", "2": "I'll give you 2,000 to fix his car.", "3": "Hello."},
    ]
    def infer(prompt, **kwargs):
        calls.append((prompt, kwargs["log_title"]))
        answer = replies.pop(0)
        assert kwargs["valid_def"](answer)["status"] == "success"
        return answer
    monkeypatch.setattr(local_translation, "ask_gpt", infer)
    monkeypatch.setattr(local_translation, "effective_target_language", lambda _: "en")
    source = "明天我出两千给你修好我的车。\n明天给你两千修好他的车。\n你好。"
    translated, original = local_translation.translate_localized(source, "UNTRUSTED_ROLE_HYPOTHESIS")
    assert translated.splitlines() == ["I'll pay 2,000 to fix my car.", "You'll receive 2,000 to fix his car.", "Hello."]
    assert original == source
    assert [title for _, title in calls] == ["translate_localized"]
    receipt = json.loads((tmp_path / "output/log/translation_fact_normalizations.json").read_text())
    assert len(receipt["changes"]) == 2
    assert receipt["accepted"] is False


def test_unresolved_source_possessive_is_never_published(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    replies = [{"1": "Fix that car."}, {"1": "Fix the car."}]
    monkeypatch.setattr(local_translation, "ask_gpt", lambda *a, **kw: replies.pop(0))
    monkeypatch.setattr(local_translation, "effective_target_language", lambda _: "en")
    with pytest.raises(ValueError, match="explicit_possession"):
        local_translation.translate_localized("修好我的车。", "")


def test_source_fact_retry_keeps_the_affected_spoken_budget_and_explains_the_missing_fact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_translation, 'effective_target_language', lambda _: 'en')
    budgets = [{'unit_id': f'U{i}', 'target_speech_seconds': seconds,
                'max_natural_speech_seconds': seconds * 1.08, 'max_speed_factor': 1.08,
                'target_max_words': 5} for i, seconds in enumerate([9., 1.23, 8.])]
    def infer(prompt, **kwargs):
        if kwargs['log_title'] == 'translate_localized':
            return {'1': 'Hello.', '2': 'Go out.', '3': 'Goodbye.'}
        assert kwargs['log_title'] == 'translate_localized_factual'
        assert '目标 1.230 秒' in prompt and '目标 9.000 秒' not in prompt
        assert 'leave town / leave the city' in prompt and 'CPS' not in prompt
        retry = kwargs['validation_retry_prompt'](prompt, {'1': 'Go out.'}, 'location omitted')
        assert 'rejected_draft' in retry and 'location omitted' in retry and '目标 1.230 秒' in retry
        answer = {'1': 'Leave town.'}
        assert kwargs['valid_def'](answer)['status'] == 'success'
        return answer
    monkeypatch.setattr(local_translation, 'ask_gpt', infer)
    translated, _ = local_translation.translate_localized('你好。\n出城。\n再见。', '', speech_budgets=budgets)
    assert translated.splitlines() == ['Hello.', 'Leave town.', 'Goodbye.']


def test_translation_language_names_preserve_locales_and_freeform():
    assert translation_language_name("zh_TW") == "繁體中文"
    assert translation_language_name("zh") == "简体中文"
    assert translation_language_name("en-US") == "英语 (en-US)"
    assert translation_language_name("vi") == "越南语"
    assert translation_language_name("Brazilian Portuguese, informal") == "Brazilian Portuguese, informal"


def test_review_batches_keep_context_and_glossary(tmp_path, monkeypatch):
    settings = {"target_language": "en", "translation.review_batch_max_lines": 2, "translation.review_context_lines": 1}
    monkeypatch.setattr(translation_review, "load_key", lambda k, default=None: settings.get(k, default))
    prompts = []
    def fake_review(prompt, **kwargs):
        prompts.append(prompt)
        return {"issues": []}
    monkeypatch.setattr(translation_review, "ask_gpt", fake_review)
    rows = pd.DataFrame([{"LineID": f"L{i}", "Speaker": "S1", "Source": f"源{i}", "Translation": f"Line {i}"} for i in range(5)])
    _, report = translation_review.review_and_repair_translation(
        rows, {"terms": [{"src": "专有名词", "tgt": "ProperName"}]}, tmp_path / "review.json")
    assert report["status"] == "pass"
    assert report["review_batch_count"] == 3
    assert report["review_basis"] == "same_model_self_review"
    assert all("ProperName" in prompt for prompt in prompts)
    assert '"line_id": "L2"' in prompts[0]  # read-only lookahead
    assert '"line_id": "L1"' in prompts[1]  # read-only previous turn


def test_review_rejects_fix_to_neighboring_batch(monkeypatch):
    monkeypatch.setattr(translation_review, "load_key", lambda k, default=None: 1 if k == "translation.review_batch_max_lines" else default)
    monkeypatch.setattr(translation_review, "ask_gpt", lambda *a, **kw: {"issues": [{"line_id": "L2", "severity": "critical", "suggested_fix": "changed"}]})
    with pytest.raises(ValueError, match="outside focused lines"):
        translation_review.review_in_batches([{"line_id": "L1"}, {"line_id": "L2"}], "", "", {}, "test")


def test_translation_manifest_detects_strategy_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(translation_state, "get_translation_strategy", lambda _: "localized")
    translation_state.record_llm_stage("translate")
    monkeypatch.setattr(translation_state, "get_translation_strategy", lambda _: "legacy")
    assert "manifest_provider_mismatch" in translation_state.build_translation_status()["stages"]["translate"]["reasons"]


def test_pipeline_manifest_records_actual_local_model(tmp_path, monkeypatch):
    from core.pipeline.runner import PipelineRun, _record_step_manifest
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output/log"
    output.mkdir(parents=True)
    (output / "translation_results.xlsx").write_bytes(b"test artifact")
    _record_step_manifest("translate", PipelineRun("test", "subtitle", "zh", "en", ["translate"]))
    manifest = json.loads((output / "translation_results.xlsx.manifest.json").read_text())
    assert manifest["models"]["llm_provider"] == "omlx"
    assert "Index-Translate" in manifest["models"]["llm"]
    assert "core/local_translation.py" in manifest["step_version"]["code_paths"]
def test_dialogue_rendering_removes_generated_emphasis_but_preserves_literal_source_stars():
    from core.local_translation import clean_dialogue_rendering
    assert clean_dialogue_rendering('这就是惊喜', "Now *that's* a surprise.", 'en')[0] == "Now that's a surprise."
    assert clean_dialogue_rendering('星号 * 是通配符', 'The * is a wildcard.', 'en') == ('The * is a wildcard.', [])


def test_large_number_notation_keeps_the_exact_value_and_currency_unit():
    from core.local_translation import clean_dialogue_rendering
    assert clean_dialogue_rendering('一百八十万两银子', 'You will receive one million eight hundred thousand taels of silver.', 'en')[0] == 'You will receive 1.8 million taels of silver.'
    assert clean_dialogue_rendering('两千零五', 'Pay two thousand and five.', 'en')[0] == 'Pay 2,005.'
    assert clean_dialogue_rendering('三天之后', 'In three days.', 'en') == ('In three days.', [])
