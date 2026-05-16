from core.llm_provider import LLMProviderConfig, build_completion_args, create_chat_client, fix_base_url
from core.providers.contracts import TTSRequest
from core.providers.mlx_tts import MlxTTSRouter, looks_vietnamese
from core.providers.quality import content_similarity, reference_leak_score


def test_fix_base_url_adds_v1_for_openai_compatible_roots():
    assert fix_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
    assert fix_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"


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


def test_mlx_router_routes_vietnamese_to_indextts2():
    req = TTSRequest(
        text="Xin chào, bạn có khỏe không?",
        output_path="output/audio/segs/test.wav",
        target_language="vi",
    )
    assert looks_vietnamese(req.text)
    assert MlxTTSRouter().select_backend(req) == "indextts2"


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


def test_quality_text_scores_are_bounded():
    assert content_similarity("Xin chao", "Xin chao") == 1.0
    assert 0.0 <= reference_leak_score("hello reference", "hello target") <= 1.0


def test_reference_leak_ignores_shared_numbers_for_chinese_source():
    assert reference_leak_score("优惠 200 元，价格 329", "Giảm 200, giá 329") == 0.0
    assert reference_leak_score("优惠 200 元", "优惠 200") > 0


def test_content_similarity_normalizes_vietnamese_number_words():
    assert content_similarity("mười chín, mười tám, mười bảy", "19, 18, 17") == 1.0
    assert content_similarity("Mười hai, mười một, mười", "12, 11, 10") == 1.0
