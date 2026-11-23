import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from threading import Lock
import json_repair
import json
import time
import hashlib
from requests.exceptions import RequestException
from core.config_utils import load_key
from core.providers.omlx_memory import omlx_inference_phase
from core.llm_provider import build_completion_args, completion_fingerprint, create_chat_client, get_llm_provider_config

JSON_REPAIR_DECODE_ERROR = getattr(json_repair, "JSONDecodeError", json.JSONDecodeError)

LOG_FOLDER = 'output/gpt_log'
LOCK = Lock()


class IncompleteLLMResponseError(ValueError):
    """An identical retry cannot complete output cut off by a hard budget."""


class RepeatedInvalidLLMResponseError(ValueError):
    """A corrective request returned an already rejected result unchanged."""


def is_non_retryable_api_error(error):
    """Return true for client/account errors that another identical request cannot fix."""
    return isinstance(error, (IncompleteLLMResponseError, RepeatedInvalidLLMResponseError)) or getattr(error, "status_code", None) in {400, 401, 402, 403, 404, 422}

def save_log(model, prompt, response, log_title = 'default', message = None, request_fingerprint=None, metadata=None):
    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_data = {
        "model": model,
        "prompt": prompt,
        "response": response,
        "message": message,
        "request_fingerprint": request_fingerprint,
        "metadata": metadata,
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

def check_ask_gpt_history(prompt, model, log_title, request_fingerprint=None):
    # check if the prompt has been asked before
    if not os.path.exists(LOG_FOLDER):
        return False
    file_path = os.path.join(LOG_FOLDER, f"{log_title}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in reversed(data):
                if item.get("prompt") == prompt and item.get("model") == model:
                    if request_fingerprint is not None and item.get("request_fingerprint") != request_fingerprint:
                        continue
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
    response_normalizer=None,
    validation_retry_prompt=None,
    provider_override=None,
):
    if provider_override is None:
        from core.llm_routes import routed_provider_name
        routed = routed_provider_name(log_title)
        provider = get_llm_provider_config(routed) if routed else get_llm_provider_config()
    else:
        provider = provider_override
    if timeout_seconds is not None:
        provider = replace(provider, timeout_seconds=float(timeout_seconds))
    request_fingerprint = completion_fingerprint(provider, response_json)
    if validation_retry_prompt is not None:
        request_fingerprint += "|validation_retry=" + validation_retry_prompt.__qualname__
    with LOCK:
        history_response = check_ask_gpt_history(prompt, provider.model, log_title, request_fingerprint)
    # Validators may call the model for semantic/display checks. Never hold
    # the log lock across them: a cached translation otherwise deadlocks on
    # the nested request's cache read.
    cached_error = None
    if history_response:
        if response_normalizer is not None:
            history_response = response_normalizer(history_response)
        if valid_def:
            valid_response = valid_def(history_response)
            if not isinstance(valid_response, dict) or valid_response.get("status") != "success":
                cached_error = valid_response.get("message", "Invalid cached response") if isinstance(valid_response, dict) else "Invalid cached response"
                with LOCK:
                    save_log(provider.model, prompt, history_response, log_title="error", message=cached_error)
        if cached_error is None:
            return history_response
    messages = [{"role": "user", "content": prompt}]
    if cached_error:
        if validation_retry_prompt is not None:
            messages = [{"role": "user", "content": validation_retry_prompt(prompt, history_response, cached_error)}]
        else:
            messages += [{'role': 'assistant', 'content': json.dumps(history_response, ensure_ascii=False)},
                         {'role': 'user', 'content': 'Correct the previous response. Validation: ' + cached_error}]
    client = create_chat_client(provider)
    rejected_results = set()

    def reject_response(data, message, metadata):
        metadata['outcome'] = 'validation_rejected'
        metadata['error'] = message
        identity = json.dumps({'response': data, 'reason': message}, ensure_ascii=False, sort_keys=True)
        repeated = identity in rejected_results
        rejected_results.add(identity)
        with LOCK:
            save_log(provider.model, prompt, data, log_title='error', message=message,
                     request_fingerprint=request_fingerprint,
                     metadata={**metadata, 'log_title': log_title,
                               'request_messages': [dict(m) for m in messages],
                               'repeated_invalid_response': repeated})
        if repeated:
            raise RepeatedInvalidLLMResponseError(
                'Validation still failed: corrective request repeated the same rejected response: ' + message)

    if max_retries is None:
        try:
            max_retries = load_key("api.retry_attempts")
        except KeyError:
            max_retries = 3

    # Fast text failures get one repair request at most, including transport retries.
    if load_key("quality_mode", "high_sync") == "fast":
        max_retries = min(int(max_retries), 2)

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
        delay = 0 if provider.name.startswith("omlx") or provider.provider_type=="mlx_local" else calculate_delay(prompt)
        time.sleep(delay)  # 根据 TPM 限制动态调整延迟
        response_metadata = {'provider': provider.name, 'base_url': provider.base_url,
                             'attempt': attempt + 1, 'max_attempts': max_retries,
                             'outcome': 'request_error', 'log_title': log_title,
                             'request_messages_sha256': hashlib.sha256(
                                 json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        request_started = time.monotonic()
        try:
            completion_args = build_completion_args(provider, messages, response_json)
            request_started = time.monotonic()
            with omlx_inference_phase(provider):
                response = client.chat.completions.create(**completion_args, timeout=provider.timeout_seconds)
            response_metadata['outcome'] = 'response_error'
            usage = getattr(response, "usage", None)
            response_metadata.update({
                "elapsed_seconds": round(time.monotonic() - request_started, 3),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "usage": usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None,
            })
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) not in (None, "stop"):
                raise IncompleteLLMResponseError(
                    f"Incomplete LLM response ({choice.finish_reason}); reduce the batch size "
                    "or increase the provider max_tokens. Refusing to repair truncated JSON."
                )
            if not isinstance(choice.message.content, str) or not choice.message.content.strip():
                raise ValueError("LLM returned no text content")

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
                    if response_normalizer is not None:
                        original_data = response_data
                        response_data = response_normalizer(response_data)
                        if response_data != original_data:
                            response_metadata['response_normalization'] = {
                                'policy': response_normalizer.__name__, 'original_parsed_response': original_data}

                    # check if the response is valid, otherwise save the log and raise error and retry
                    if valid_def:
                        valid_response = valid_def(response_data)
                        if valid_response['status'] != 'success':
                            reject_response(response_data, valid_response['message'], response_metadata)
                            raise ValueError(f"❎ API response error: {valid_response['message']}")

                    response_metadata['outcome'] = 'accepted'
                    break  # Successfully accessed and parsed, break the loop
                except JSON_REPAIR_DECODE_ERROR as e:
                    # Actual JSON parsing failure
                    response_data = response.choices[0].message.content
                    print(f"❎ JSON parsing failed. Retrying: '''{response_data[:200]}...'''")
                    reject_response(response_data, f"JSON parsing failed: {str(e)}", response_metadata)
                    if attempt == max_retries - 1:
                        raise Exception(f"JSON parsing still failed after {max_retries} attempts: {e}\n Please check your network connection or API key or `output/gpt_log/error.json` to debug.")
                except RepeatedInvalidLLMResponseError:
                    raise
                except ValueError as e:
                    # Validation failure (e.g., translation too long)
                    print(f"❎ Validation failed: {e}. Retrying...")
                    if attempt == max_retries - 1:
                        raise Exception(f"Validation still failed after {max_retries} attempts: {e}")
                    # The corrective request must contain the rejected draft.
                    # Without it, deterministic local models repeatedly answer
                    # the original prompt while having no text to correct.
                    if validation_retry_prompt is not None:
                        # Specialized translators keep their native single-user
                        # source/constraints format on every corrective request.
                        retry_prompt = validation_retry_prompt(prompt, response_data, str(e))
                        messages = [{"role":"user", "content":retry_prompt}]
                    else:
                        messages.append({"role": "assistant", "content": raw_content})
                        messages.append({"role": "user", "content": (
                            f"The previous response failed validation: {e}. "
                            "Return a corrected JSON response for the original input, "
                            "following the required keys and value types exactly."
                        )})
                except Exception as e:
                    response_data = response.choices[0].message.content
                    print(f"❎ Error processing response: {e}. Retrying...")
                    save_log(provider.model, prompt, response_data, log_title="error", message=f"Error: {str(e)}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Still failed after {max_retries} attempts: {e}\n Please check `output/gpt_log/error.json` to debug.")
            else:
                response_data = response.choices[0].message.content
                if valid_def:
                    valid_response = valid_def(response_data)
                    if valid_response["status"] != "success":
                        reject_response(response_data, valid_response['message'], response_metadata)
                        if validation_retry_prompt is not None:
                            messages = [{"role": "user", "content": validation_retry_prompt(
                                prompt, response_data, valid_response["message"])}]
                        else:
                            messages += [{"role": "assistant", "content": response_data},
                                         {"role": "user", "content": (
                                             "The previous response failed validation: " + valid_response["message"]
                                             + ". Correct it for the original input. Return only the requested text.")}]
                        raise ValueError(valid_response["message"])
                response_metadata['outcome'] = 'accepted'
                break  # Non-JSON format, break the loop directly

        except Exception as e:
            response_metadata.setdefault('error', f'{type(e).__name__}: {e}')
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
        finally:
            # One lightweight event for every actual request, including failed
            # validation and transport. Cache hits do not masquerade as calls.
            response_metadata.setdefault('elapsed_seconds', round(time.monotonic() - request_started, 3))
            with LOCK:
                os.makedirs(LOG_FOLDER, exist_ok=True)
                with open(os.path.join(LOG_FOLDER, 'request_events.jsonl'), 'a', encoding='utf-8') as event_file:
                    event_file.write(json.dumps(response_metadata, ensure_ascii=False) + '\n')
    with LOCK:
        if log_title != 'None':
            save_log(provider.model, prompt, response_data, log_title=log_title, request_fingerprint=request_fingerprint, metadata=response_metadata)

    return response_data


if __name__ == '__main__':
    print(ask_gpt('hi there hey response in json format, just return 200.' , response_json=True, log_title=None))
