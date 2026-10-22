import json

import pandas as pd
import pytest

from core import dialogue_roles as roles
from core.translation_context import format_dialogue, split_speaker_aware_batches
from core.step4_2_translate_all import assemble_translation_results
from core.translation_review import is_valid_target_language


def scene():
    return pd.DataFrame([
        {"line_id": "L00001", "text": "三天后我出一百八十万。", "speaker": "S01", "start": 0., "end": 3.},
        {"line_id": "L00002", "text": "他说三天后给你一百八十万。", "speaker": "S01", "start": 4., "end": 7.},
        {"line_id": "L00003", "text": "对，我出钱。", "speaker": "S03", "start": 8., "end": 9.},
    ])


def cards():
    return [
        {"character_id": "C01", "label": "付款人", "evidence": [{"line_id": "L00001", "quote": "我出一百八十万"}]},
        {"character_id": "C02", "label": "转述人", "evidence": [{"line_id": "L00002", "quote": "他说"}]},
    ]


def response():
    return {"characters": [], "assignments": [
        {"line_id": "L00001", "character_id": "C01", "confidence": "high", "action_agent": "C01",
         "evidence": [{"line_id": "L00001", "quote": "我出一百八十万"}]},
        {"line_id": "L00002", "character_id": "C02", "confidence": "high", "relays_character": "C01", "action_agent": "C01",
         "evidence": [{"line_id": "L00001", "quote": "我出一百八十万"}, {"line_id": "L00002", "quote": "他说"}]},
        {"line_id": "L00003", "character_id": "C01", "confidence": "medium", "action_agent": "C01",
         "evidence": [{"line_id": "L00003", "quote": "对，我出钱"}]},
    ]}


def test_semantic_roles_can_split_one_voice_cluster_and_join_different_clusters(tmp_path, monkeypatch):
    rows = scene()
    def infer(prompt, **kwargs):
        assert '"acoustic_speaker"' not in prompt
        assert '"S01"' not in prompt
        return response()
    monkeypatch.setattr(roles, "ask_gpt", infer)
    summary = roles.reconcile_dialogue_roles({"characters": cards()}, rows, {"groups": []}, tmp_path / "roles.json")
    attached = roles.attach_dialogue_roles(rows, summary)
    assert attached.character_id.tolist() == ["C01", "C02", "C01"]
    assert attached.relays_character.tolist() == ["unknown", "C01", "unknown"]
    assert attached.action_agent.tolist() == ["C01", "C01", "C01"]
    pd.testing.assert_frame_equal(attached[rows.columns], rows)  # no timing/text/voice mutation
    report = json.loads((tmp_path / "roles.json").read_text())
    assert report["acoustic_to_characters"]["S01"] == {"C01": 1, "C02": 1}
    assert report["human_quality"] == "pending"
    assert "[character=C02]" in format_dialogue(attached)
    assert "S01" not in format_dialogue(attached)
    batches = split_speaker_aware_batches(attached)
    output = assemble_translation_results([(0, ["I'll pay.", "He'll pay.", "I'll pay."])], batches, attached)
    assert output.Character.tolist() == ["C01", "C02", "C01"]
    assert output.Speaker.tolist() == ["S01", "S01", "S03"]
    assert output.RelaysCharacter.tolist()[1] == "C01"


@pytest.mark.parametrize("change", ["low_confidence", "invented_evidence", "unknown_character"])
def test_unreliable_assignment_is_not_promoted_to_certain_role(change):
    rows = {r["line_id"]: r for r in roles.dialogue_records(scene())}
    item = response()["assignments"][1]
    if change == "low_confidence":
        item["confidence"] = "low"
    elif change == "invented_evidence":
        item["evidence"] = [{"line_id": "L00002", "quote": "我是付款人"}]
    else:
        item["character_id"] = "C99"
    result = roles.ground_assignment(item, rows["L00002"], cards(), rows)
    assert result["character_id"] == "unknown"
    assert result["action_agent"] == "unknown"
    assert result["acoustic_speaker"] == "S01"
    assert result["decision"] == "needs_review"


def test_role_map_cannot_follow_stale_line_ids_after_source_changes(tmp_path, monkeypatch):
    rows = scene()
    monkeypatch.setattr(roles, "ask_gpt", lambda *a, **kw: response())
    summary = roles.reconcile_dialogue_roles({"characters": cards()}, rows, {"groups": []}, tmp_path / "roles.json")
    rows.loc[1, "text"] = "我付钱。"
    with pytest.raises(ValueError, match="source changed"):
        roles.attach_dialogue_roles(rows, summary)


def test_every_line_is_reconciled_exactly_once():
    reply = response()
    assert roles.validate_assignments(reply, scene().line_id.tolist())["status"] == "success"
    reply["assignments"][2]["line_id"] = "L00002"
    assert roles.validate_assignments(reply, scene().line_id.tolist())["status"] == "error"


def test_complete_role_batch_extracts_focus_without_losing_or_reassigning_context():
    reply=response()
    normalized=roles.normalize_assignment_focus(reply,["L00001","L00002"],scene().line_id.tolist())
    assert [item["line_id"] for item in normalized["assignments"]] == ["L00001","L00002"]
    assert normalized["assignments"] == reply["assignments"][:2]
    assert len(reply["assignments"]) == 3
    assert roles.validate_assignments(normalized,["L00001","L00002"])["status"] == "success"


@pytest.mark.parametrize("bad_ids", [["L00001","L00003"], ["L00001","L00002","L99999"],
                                     ["L00001","L00002","L00002"], ["L00001",None]])
def test_role_focus_normalization_does_not_hide_missing_duplicate_or_invented_ids(bad_ids):
    reply={"characters":[],"assignments":[{"line_id":lid} for lid in bad_ids]}
    normalized=roles.normalize_assignment_focus(reply,["L00001","L00002"],scene().line_id.tolist())
    assert normalized == reply
    assert roles.validate_assignments(normalized,["L00001","L00002"])["status"] == "error"


def test_character_ids_cannot_leak_into_spoken_translation():
    assert not is_valid_target_language("C01 will pay for C02's leg.", "en")


def test_invented_first_person_in_a_relay_cannot_determine_the_payer():
    rows = {r["line_id"]: r for r in roles.dialogue_records(scene())}
    item = response()["assignments"][1]
    item["pronoun_referents"] = {"我": "C02", "他": "C01"}
    result = roles.ground_assignment(item, rows["L00002"], cards(), rows)
    assert result["character_id"] == "C02"
    assert result["action_agent"] == "unknown"
    assert "pronoun_not_spoken:我" in result["consistency_conflicts"]
    assert "我" not in result["pronoun_referents"]
    assert result["relations_status"] == "needs_review"


def test_relay_with_an_unknown_original_speaker_is_review_required():
    rows = {r["line_id"]: r for r in roles.dialogue_records(scene())}
    item = response()["assignments"][1]
    item.update(speech_act="转述承诺", relays_character="unknown")
    result = roles.ground_assignment(item, rows["L00002"], cards(), rows)
    assert result["decision"] == "applied_inferred"
    assert result["relations_status"] == "needs_review"
    assert result["action_agent"] == "unknown"
    assert "relayed_actor_unresolved" in result["consistency_conflicts"]


@pytest.mark.parametrize("unit", ["", "两银子"])
def test_repeated_promise_does_not_collapse_relayer_and_payer(unit):
    records = [{"line_id": "L1", "text": "明天我出两千修好我的车。"},
               {"line_id": "L2", "text": f"明天给你两千{unit}修好他的车。"}]
    assignments = [{"line_id": record["line_id"], "character_id": "C01", "consistency_conflicts": [],
                    "action_agent": "C01", "decision": "applied_inferred"} for record in records]
    result = roles.flag_repeated_promise_links(assignments, records)
    assert result[0]["character_id"] == "C01"
    assert result[1]["character_id"] == "unknown"
    assert result[1]["action_agent"] == "unknown"
    assert result[1]["relay_link"]["antecedent_line_id"] == "L1"
    assert result[1]["relay_link"]["status"] == "hypothesis"
    assert result[1]["decision"] == "needs_review"
