import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from threading import Lock
import json_repair
import json 
import time
from requests.exceptions import RequestException
from core.config_utils import load_key
from core.llm_provider import build_completion_args, create_chat_client, get_llm_provider_config

JSON_REPAIR_DECODE_ERROR = getattr(json_repair, "JSONDecodeError", json.JSONDecodeError)

LOG_FOLDER = 'output/gpt_log'
LOCK = Lock()


def is_non_retryable_api_error(error):
    """Return true for client/account errors that another identical request cannot fix."""
    return getattr(error, "status_code", None) in {400, 401, 402, 403, 404, 422}

def save_log(model, prompt, response, log_title = 'default', message = None):
    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_data = {
        "model": model,
        "prompt": prompt,
        "response": response,
        "message": message
    }
    log_file = os.path.join(LOG_FOLDER, f"{log_title}.json")
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_data)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
        
def check_ask_gpt_history(prompt, model, log_title):
    # check if the prompt has been asked before
    if not os.path.exists(LOG_FOLDER):
        return False
    file_path = os.path.join(LOG_FOLDER, f"{log_title}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item.get("prompt") == prompt and item.get("model") == model:
                    return item["response"]
    return False

def ask_gpt(
    prompt,
    response_json=True,
    valid_def=None,
    log_title='default',
    max_retries=None,
    retry_interval=None,
    timeout_seconds=None,
):
    provider = get_llm_provider_config()
    if timeout_seconds is not None:
        provider = replace(provider, timeout_seconds=float(timeout_seconds))
    with LOCK:
        history_response = check_ask_gpt_history(prompt, provider.model, log_title)
        if history_response:
            return history_response
    
    messages = [{"role": "user", "content": prompt}]
    client = create_chat_client(provider)

    if max_retries is None:
        try:
            max_retries = load_key("api.retry_attempts")
        except KeyError:
            max_retries = 3

    if retry_interval is None:
        try:
            retry_interval = load_key("api.retry_interval")
        except KeyError:
            retry_interval = 60

    # 移除固定的 time.sleep(1)，改为动态延迟
    def calculate_delay(prompt):
        try:
            tpm_limit = load_key("api.tpm_limit")
        except KeyError:
            tpm_limit = 10000
        token_count = len(prompt.split())  # 粗略估计请求中的 token 数量
        delay = 60 / (tpm_limit / token_count) if tpm_limit > 0 and token_count > 0 else 0  # 计算所需延迟
        return delay

    for attempt in range(max_retries):
        delay = calculate_delay(prompt)
        time.sleep(delay)  # 根据 TPM 限制动态调整延迟
        try:
            completion_args = build_completion_args(provider, messages, response_json)
            response = client.chat.completions.create(**completion_args)
            
            if response_json:
                try:
                    raw_content = response.choices[0].message.content
                    # Strip markdown code blocks if present (```json ... ``` or ``` ... ```)
                    content = raw_content.strip()
                    # Check for markdown code block wrapper
                    if content.startswith('```'):
                        # Find the end of the code block
                        end_marker = content.rfind('```')
                        if end_marker > 3:  # There is an end marker
                            # Remove first line (```json or ```) and last ```
                            first_newline = content.find('\n')
                            if first_newline != -1:
                                content = content[first_newline+1:end_marker].strip()
                    
                    response_data = json_repair.loads(content)
                    
                    # check if the response is valid, otherwise save the log and raise error and retry
                    if valid_def:
                        valid_response = valid_def(response_data)
                        if valid_response['status'] != 'success':
                            save_log(provider.model, prompt, response_data, log_title="error", message=valid_response['message'])
                            raise ValueError(f"❎ API response error: {valid_response['message']}")
                        
                    break  # Successfully accessed and parsed, break the loop
                except JSON_REPAIR_DECODE_ERROR as e:
                    # Actual JSON parsing failure
                    response_data = response.choices[0].message.content
                    print(f"❎ JSON parsing failed. Retrying: '''{response_data[:200]}...'''")
                    save_log(provider.model, prompt, response_data, log_title="error", message=f"JSON parsing failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise Exception(f"JSON parsing still failed after {max_retries} attempts: {e}\n Please check your network connection or API key or `output/gpt_log/error.json` to debug.")
                except ValueError as e:
                    # Validation failure (e.g., translation too long)
                    print(f"❎ Validation failed: {e}. Retrying...")
                    if attempt == max_retries - 1:
                        raise Exception(f"Validation still failed after {max_retries} attempts: {e}")
                except Exception as e:
                    response_data = response.choices[0].message.content
                    print(f"❎ Error processing response: {e}. Retrying...")
                    save_log(provider.model, prompt, response_data, log_title="error", message=f"Error: {str(e)}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Still failed after {max_retries} attempts: {e}\n Please check `output/gpt_log/error.json` to debug.")
            else:
                response_data = response.choices[0].message.content
                break  # Non-JSON format, break the loop directly
                
        except Exception as e:
            if is_non_retryable_api_error(e):
                raise
            if attempt < max_retries - 1:
                if isinstance(e, RequestException):
                    print(f"Request error: {e}. Retrying ({attempt + 1}/{max_retries})...")
                else:
                    print(f"Unexpected error occurred: {e}\nRetrying...")
                time.sleep(min(retry_interval, 2 ** attempt))
            else:
                raise Exception(f"Still failed after {max_retries} attempts: {e}")
    with LOCK:
        if log_title != 'None':
            save_log(provider.model, prompt, response_data, log_title=log_title)

    return response_data


if __name__ == '__main__':
    print(ask_gpt('hi there hey response in json format, just return 200.' , response_json=True, log_title=None))
