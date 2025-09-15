from ruamel.yaml import YAML
from typing import Any
import os, sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = 'config.yaml'
config_lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True

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
        
        with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
            yaml.dump(data, file)
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