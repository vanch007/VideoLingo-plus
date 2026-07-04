from ruamel.yaml import YAML
from typing import Any
import os, sys
import threading
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = 'config.yaml'
config_lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True

ENV_OVERRIDES = {
    "hf_token": ("VIDEOLINGO_HF_TOKEN", "HF_TOKEN"),
    "api.key": ("VIDEOLINGO_API_KEY", "OPENAI_API_KEY"),
    "api.base_url": ("VIDEOLINGO_API_BASE_URL",),
    "api.model": ("VIDEOLINGO_API_MODEL",),
    "llm.providers.omlx.api_key": ("OMLX_API_KEY", "VIDEOLINGO_OMLX_API_KEY"),
    "llm.providers.omlx.base_url": ("OMLX_BASE_URL", "VIDEOLINGO_OMLX_BASE_URL"),
    "llm.providers.omlx.model": ("OMLX_MODEL", "VIDEOLINGO_OMLX_MODEL"),
    "openai_tts.api_key": ("VIDEOLINGO_OPENAI_API_KEY", "OPENAI_API_KEY"),
    "elevenlabs_tts.api_key": ("VIDEOLINGO_ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY"),
    "elevenlabs.api_key": ("VIDEOLINGO_ELEVENLABS_API_KEY", "ELEVENLABS_API_KEY"),
}

SENSITIVE_KEYS = {
    "hf_token",
    "api.key",
    "llm.providers.omlx.api_key",
    "openai_tts.api_key",
    "elevenlabs_tts.api_key",
    "elevenlabs.api_key",
}


def _env_value(key_path: str):
    for env_name in ENV_OVERRIDES.get(key_path, ()):
        value = os.environ.get(env_name)
        if value:
            return value
    return None

def load_config():
    """Load the config.yaml file into a dictionary."""
    with config_lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            return yaml.load(file)

def load_key(key_path: str, default=None):
    """Load a specific key's value from the config.yaml file.
    
    Args:
        key_path (str): Path to the key in dot notation (e.g., 'whisper.language')
        default: Default value to return if key is not found
        
    Returns:
        The value of the key or default if not found
    """
    env_override = _env_value(key_path)
    if env_override is not None:
        return env_override

    config = load_config()
    keys = key_path.split('.')
    value = config
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        if default is not None:
            return default
        raise KeyError(f"Key '{key_path}' not found in config")


def get_env_names(key_path: str) -> tuple[str, ...]:
    return ENV_OVERRIDES.get(key_path, ())


def is_sensitive_key(key_path: str) -> bool:
    return key_path in SENSITIVE_KEYS

def update_key(key: str, new_value: Any) -> bool:
    with config_lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        keys = key.split('.')
        current = data
        for i, k in enumerate(keys[:-1]):
            if k not in current or not isinstance(current.get(k), dict):
                current[k] = {}
            current = current[k]

        current[keys[-1]] = new_value

        config_dir = os.path.dirname(os.path.abspath(CONFIG_PATH)) or "."
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                'w',
                encoding='utf-8',
                dir=config_dir,
                prefix='.config.',
                suffix='.tmp',
                delete=False,
            ) as file:
                temp_path = file.name
                yaml.dump(data, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, CONFIG_PATH)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        return True
        
# basic utils
def get_joiner(language):
    if language in load_key('language_split_with_space'):
        return " "
    elif language in load_key('language_split_without_space'):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")

def get_work_dir():
    return os.getcwd()

if __name__ == "__main__":
    print(load_key('language_split_with_space'))
