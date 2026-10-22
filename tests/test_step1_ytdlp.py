import pytest
from unittest.mock import MagicMock, patch
from core.step1_ytdlp import (
    extract_clean_url,
    download_video_ytdlp,
    sanitize_filename,
    _browser_cookie_tuple,
)

def test_sanitize_filename():
    assert sanitize_filename("hello/world:test?") == "helloworldtest"
    assert sanitize_filename("   .my video.   ") == "my video"
    assert sanitize_filename("") == "video"

def test_extract_clean_url():
    assert extract_clean_url("https://www.youtube.com/watch?v=123") == "https://www.youtube.com/watch?v=123"
    assert extract_clean_url("  https://youtu.be/abc?t=10  ") == "https://youtu.be/abc?t=10"
    douyin_share = "8.12 复制打开抖音，看看【测试账号】的作品 https://v.douyin.com/iabc123/ 精彩不断"
    assert extract_clean_url(douyin_share) == "https://v.douyin.com/iabc123/"
    bili_share = "【高清精选】视频分享 https://www.bilibili.com/video/BV1xx411c7mD?spm_id=1 推荐观看"
    assert extract_clean_url(bili_share) == "https://www.bilibili.com/video/BV1xx411c7mD?spm_id=1"
    assert extract_clean_url("") == ""

def test_browser_cookie_tuple():
    assert _browser_cookie_tuple("chrome") == ("chrome", None, None, None)
    assert _browser_cookie_tuple("chrome:Profile 1") == ("chrome", "Profile 1", None, None)
    assert _browser_cookie_tuple(None) is None

def test_download_video_ytdlp_options_youtube(tmp_path, monkeypatch):
    captured_opts = {}
    captured_urls = []
    class MockYDL:
        def __init__(self, opts):
            nonlocal captured_opts
            captured_opts = opts
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def download(self, urls):
            nonlocal captured_urls
            captured_urls = urls
    monkeypatch.setattr("core.step1_ytdlp.ensure_canonical_source_video", lambda _: "output/source.mp4")
    monkeypatch.setattr("yt_dlp.YoutubeDL", MockYDL)
    download_video_ytdlp(
        "https://www.youtube.com/watch?v=test123",
        save_path=str(tmp_path),
        resolution="1080",
    )
    assert captured_urls == ["https://www.youtube.com/watch?v=test123"]
    assert "bestvideo[height<=1080]+bestaudio" in captured_opts["format"]
    assert captured_opts["merge_output_format"] == "mp4"
    assert captured_opts["noplaylist"] is True
    assert captured_opts["retries"] == 5

def test_download_video_ytdlp_options_bilibili_headers(tmp_path, monkeypatch):
    captured_opts = {}
    class MockYDL:
        def __init__(self, opts):
            nonlocal captured_opts
            captured_opts = opts
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def download(self, urls):
            pass
    monkeypatch.setattr("core.step1_ytdlp.ensure_canonical_source_video", lambda _: "output/source.mp4")
    monkeypatch.setattr("yt_dlp.YoutubeDL", MockYDL)
    download_video_ytdlp(
        "【测试】 https://www.bilibili.com/video/BV1xx411c7mD",
        save_path=str(tmp_path),
        resolution="best",
    )
    assert captured_opts["format"] == "bestvideo+bestaudio/best"
    assert captured_opts["http_headers"].get("Referer") == "https://www.bilibili.com/"

def test_download_video_ytdlp_douyin_shortlink_resolved(tmp_path, monkeypatch):
    captured_urls = []
    class MockYDL:
        def __init__(self, opts):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def download(self, urls):
            nonlocal captured_urls
            captured_urls = urls
    monkeypatch.setattr("core.step1_ytdlp.ensure_canonical_source_video", lambda _: "output/source.mp4")
    monkeypatch.setattr("core.step1_ytdlp.resolve_redirect_url", lambda u: "https://www.douyin.com/video/7123456789012345678")
    monkeypatch.setattr("yt_dlp.YoutubeDL", MockYDL)
    download_video_ytdlp(
        "来看视频 https://v.douyin.com/abc123/ 复制打开",
        save_path=str(tmp_path),
        resolution="720",
    )
    assert captured_urls == ["https://www.douyin.com/video/7123456789012345678"]
