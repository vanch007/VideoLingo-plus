import os
import shutil
import pytest
import pandas as pd
from pathlib import Path
from batch.utils.video_processor import normalize_language_code
from batch.utils.settings_check import check_settings

def test_normalize_language_code():
    assert normalize_language_code('简体中文') == 'zh'
    assert normalize_language_code('中文') == 'zh'
    assert normalize_language_code('英语') == 'en'
    assert normalize_language_code('en') == 'en'
    assert normalize_language_code('zh') == 'zh'
    assert normalize_language_code('vi') == 'vi'
    assert normalize_language_code('越南语') == 'vi'
    assert normalize_language_code(None, 'auto') == 'auto'
    assert normalize_language_code('', 'auto') == 'auto'

def test_batch_settings_check_with_real_inputs():
    assert check_settings() is True
