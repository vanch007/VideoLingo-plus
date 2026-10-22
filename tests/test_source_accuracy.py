from core.source_accuracy import evaluate_source_checkpoints, closest_reference_span


def test_boundary_shift_does_not_become_a_lexical_omission():
    result = {"segments": [{"start": 9.9, "end": 9.9, "text": "难"},
                           {"start": 10., "end": 11., "text": "道你听不懂什么叫惊喜"},
                           {"start": 11.1, "end": 11.2, "text": "我"}]}
    reference = {"scope": "one excerpt", "checkpoints": [{"start": 10., "end": 11.,
                 "text": "难道你听不懂什么叫惊喜", "required_facts": ["惊喜"]}]}
    report = evaluate_source_checkpoints(result, reference)
    assert report["character_errors"] == 0
    assert report["checkpoints"][0]["observed"] != report["checkpoints"][0]["matched_text"]
    assert report["status"] == "pass"


def test_one_wrong_pronoun_fails_even_when_overall_character_accuracy_is_high():
    result = {"segments": [{"start": 10., "end": 11., "text": "我出一百八十万给你们出城剿匪接上他的腿"}]}
    reference = {"scope": "one core negotiation excerpt", "max_cer": .1,
                 "checkpoints": [{"start": 10., "end": 11., "text": "我出一百八十万给你们出城剿匪接上我的腿", "required_facts": ["我的腿"]}]}
    report = evaluate_source_checkpoints(result, reference)
    assert report["cer"] < .1
    assert report["critical_fact_failures"] == 1
    assert report["status"] == "fail"


def test_reference_span_alignment_still_counts_internal_hallucinated_text():
    distance, observed = closest_reference_span("我剿匪", "他说我帮他们剿匪你听到了吗")
    assert distance > 0
    assert observed != "我剿匪"
