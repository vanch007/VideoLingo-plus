import pytest

from core import llm_routes


def test_log_titles_map_to_routes():
    assert llm_routes.route_for("summary") == "analysis"
    assert llm_routes.route_for("dialogue_roles_b3") == "analysis"
    assert llm_routes.route_for("constraint_review") == "review"
    assert llm_routes.route_for("translation_semantic_review_r1_b2") == "review"
    assert llm_routes.route_for("translate_localized") is None
    assert llm_routes.route_for("sentence_splitbymeaning") is None


def test_unset_route_keeps_default_provider_and_set_route_is_used(monkeypatch):
    config = {"llm.routes.analysis": "", "llm.routes.review": "deepseek_v4_flash",
              "llm.providers.deepseek_v4_flash": {"model": "x"}}
    monkeypatch.setattr(llm_routes, "load_key", lambda key, default=None: config.get(key, default))
    assert llm_routes.routed_provider_name("summary") is None
    assert llm_routes.routed_provider_name("constraint_review") == "deepseek_v4_flash"
    assert llm_routes.routed_provider_name("translate_localized") is None
    assert llm_routes.configured_routes() == {"review": "deepseek_v4_flash"}


def test_speech_duration_translator_cannot_be_routed(monkeypatch):
    config = {"llm.routes.review": "omlx_homura", "llm.providers.omlx_homura": {"purpose": "speech_duration"}}
    monkeypatch.setattr(llm_routes, "load_key", lambda key, default=None: config.get(key, default))
    with pytest.raises(ValueError):
        llm_routes.routed_provider_name("constraint_review")


def test_ask_gpt_uses_the_routed_provider(monkeypatch):
    from core import ask_gpt as module
    asked = []
    monkeypatch.setattr(llm_routes, "routed_provider_name", lambda title: "deepseek_v4_flash" if title == "summary" else None)

    def fake_config(*args):
        asked.append(args)
        raise RuntimeError("stop")

    monkeypatch.setattr(module, "get_llm_provider_config", fake_config)
    for title in ("summary", "translate_localized"):
        with pytest.raises(RuntimeError):
            module.ask_gpt("hi", log_title=title)
    assert asked == [("deepseek_v4_flash",), ()]
