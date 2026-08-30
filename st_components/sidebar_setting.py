import os, sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st_components.imports_and_utils import ask_gpt
import streamlit as st
from core.config_utils import get_env_names, update_key, load_key
from core.all_tts_functions.tts_registry import list_selectable_tts_methods
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES

def config_input(label, key, help=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help)
    if val != load_key(key):
        update_key(key, val)
    return val

def config_number_input(label, key, help=None, min_value=None, max_value=None, step=None):
    """Generic config number input handler"""
    current_value = int(load_key(key))
    val = st.number_input(label, value=current_value, help=help, min_value=min_value, max_value=max_value, step=step)
    if val != current_value:
        update_key(key, val)
    return val

def secret_env_status(label, key):
    env_names = get_env_names(key)
    configured = any(os.environ.get(name) for name in env_names)
    status = "✅" if configured else "⚠️"
    env_text = ", ".join(env_names) if env_names else "not configured"
    st.caption(f"{status} {label}: set via env ({env_text})")
    return configured

def page_setting():

    display_language = st.selectbox("Display Language 🌐",
                                  options=list(DISPLAY_LANGUAGES.keys()),
                                  index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")))
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

    with st.expander("Doctor", expanded=False):
        st.caption("Environment and workflow readiness checks.")
        if st.button("Run Doctor", key="run_doctor"):
            from core.doctor import run_checks
            for check in run_checks(include_services=True):
                icon = "✅" if check.ok else "❌"
                st.write(f"{icon} **{check.name}**: {check.detail}")

    with st.expander(t("LLM Configuration"), expanded=True):
        providers = ["openai_compatible"] + list(load_key("llm.providers", {}).keys())
        current_provider = load_key("llm.provider", "openai_compatible")
        if current_provider not in providers:
            current_provider = "openai_compatible"
        selected_provider = st.selectbox(
            "LLM Provider",
            options=providers,
            index=providers.index(current_provider),
            help="openai_compatible uses the api block below; presets use their configured env var."
        )
        if selected_provider != current_provider:
            update_key("llm.provider", selected_provider)
            st.rerun()

        secret_env_status(t("API_KEY"), "api.key")
        if selected_provider == "openai_compatible":
            config_input(t("BASE_URL"), "api.base_url", help=t("Openai format, will add /v1/chat/completions automatically"))

            c1, c2 = st.columns([4, 1])
            with c1:
                config_input(t("MODEL"), "api.model", help=t("click to check API validity")+ " 👉")
            with c2:
                if st.button("📡", key="api"):
                    st.toast(t("API Key is valid") if check_api() else t("API Key is invalid"),
                            icon="✅" if check_api() else "❌")
        else:
            provider_cfg = load_key(f"llm.providers.{selected_provider}")
            st.caption(f"Model: `{provider_cfg['model']}`")
            st.caption(f"Base URL: `{provider_cfg['base_url']}`")
            st.caption(f"API key env: `{provider_cfg.get('api_key_env', 'n/a')}`")
            if selected_provider == "omlx":
                c_omlx1, c_omlx2 = st.columns(2)
                with c_omlx1:
                    if st.button("List oMLX Models", key="list_omlx_models"):
                        try:
                            from core.providers.omlx import list_omlx_models
                            st.json([model.__dict__ for model in list_omlx_models()], expanded=False)
                        except Exception as e:
                            st.error(f"oMLX model discovery failed: {e}")
                with c_omlx2:
                    if st.button("Smoke oMLX", key="smoke_omlx"):
                        try:
                            from core.providers.omlx import smoke_chat
                            st.json(smoke_chat(), expanded=False)
                        except Exception as e:
                            st.error(f"oMLX smoke failed: {e}")

    with st.expander(t("Subtitles Settings"), expanded=True):
        # Language codes - use t() for translated display names
        lang_codes = ["ar", "hy", "az", "be", "bs", "bg", "my", "ca", "zh", "hr", "cs", "da",
                      "nl", "en", "et", "fi", "fr", "gl", "de", "el", "gu", "he", "hi", "hu",
                      "is", "id", "it", "ja", "kn", "kk", "km", "ko", "lo", "lv", "lt", "mk",
                      "ms", "mr", "mn", "ne", "no", "fa", "pl", "pt", "ro", "ru", "sr", "sk",
                      "sl", "es", "sv", "ta", "te", "th", "tr", "uk", "ur", "vi", "cy"]

        # English names for translation keys
        lang_names = ["Arabic", "Armenian", "Azerbaijani", "Belarusian", "Bosnian", "Bulgarian",
                      "Burmese", "Catalan", "Chinese", "Croatian", "Czech", "Danish", "Dutch",
                      "English", "Estonian", "Finnish", "French", "Galician", "German", "Greek",
                      "Gujarati", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Indonesian",
                      "Italian", "Japanese", "Kannada", "Kazakh", "Khmer", "Korean", "Lao",
                      "Latvian", "Lithuanian", "Macedonian", "Malay", "Marathi", "Mongolian",
                      "Nepali", "Norwegian", "Persian", "Polish", "Portuguese", "Romanian",
                      "Russian", "Serbian", "Slovak", "Slovenian", "Spanish", "Swedish", "Tamil",
                      "Telugu", "Thai", "Turkish", "Ukrainian", "Urdu", "Vietnamese", "Welsh"]

        # Create code-to-name mapping and translated display names
        code_to_name = dict(zip(lang_codes, lang_names))
        translated_names = [t(name) for name in lang_names]
        code_to_translated = dict(zip(lang_codes, translated_names))
        translated_to_code = dict(zip(translated_names, lang_codes))

        c1, c2 = st.columns(2)
        with c1:
            # Recognition language dropdown
            current_rec_lang = load_key("whisper.language")
            rec_lang_index = lang_codes.index(current_rec_lang) if current_rec_lang in lang_codes else lang_codes.index("en")

            selected_rec = st.selectbox(
                t("Recog Lang"),
                options=translated_names,
                index=rec_lang_index
            )
            if translated_to_code[selected_rec] != current_rec_lang:
                update_key("whisper.language", translated_to_code[selected_rec])
                st.rerun()

        with c2:
            # Target language dropdown (same options as recognition)
            current_target = load_key("target_language")
            target_lang_index = lang_codes.index(current_target) if current_target in lang_codes else lang_codes.index("id")

            selected_target = st.selectbox(
                t("Target Lang"),
                options=translated_names,
                index=target_lang_index,
                help=t("Select the language for translation")
            )
            if translated_to_code[selected_target] != current_target:
                update_key("target_language", translated_to_code[selected_target])
                st.rerun()

        # ASR model selection
        asr_options = {
            "WhisperX (native word alignment)": "local",
            "stable-ts (word timestamps)": "stable-ts",
        }
        asr_display = list(asr_options.keys())
        asr_values = list(asr_options.values())
        current_runtime = load_key("whisper.runtime") if load_key("whisper.runtime") in asr_values else "local"
        current_index = asr_values.index(current_runtime)

        selected_asr = st.selectbox(
            t("ASR Model"),
            options=asr_display,
            index=current_index,
            help="Only ASR engines with validated native word timestamps can feed subtitle alignment."
        )
        runtime = asr_options[selected_asr]
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()

        if runtime == "stable-ts":
            c_sts1, c_sts2 = st.columns(2)
            with c_sts1:
                val_vad = st.slider(t("VAD Threshold"), min_value=0.0, max_value=1.0, value=float(load_key("whisper.vad_threshold", 0.3)), step=0.05, help=t("Higher value = more aggressive voice detection"))
                if val_vad != load_key("whisper.vad_threshold", 0.3):
                    update_key("whisper.vad_threshold", val_vad)
            with c_sts2:
                val_min_dur = st.slider(t("Min Word Duration"), min_value=0.0, max_value=0.5, value=float(load_key("whisper.min_word_dur", 0.1)), step=0.01, help=t("Minimum duration for a word to be kept"))
                if val_min_dur != load_key("whisper.min_word_dur", 0.1):
                    update_key("whisper.min_word_dur", val_min_dur)

            use_dq = st.toggle(t("Use Dynamic Quantization (CPU)"), value=load_key("whisper.stable_ts_dq", True), help=t("Speeds up inference on CPU, slightly lower accuracy"))
            if use_dq != load_key("whisper.stable_ts_dq", True):
                update_key("whisper.stable_ts_dq", use_dq)
                st.rerun()

        # Only show Whisper model selection for runtimes that use Whisper (local, stable-ts)
        if runtime in ["local", "stable-ts"]:
            # Dynamically determine the list of models based on the runtime and MLX setting
            help_text = ""
            whisper_models = ["medium", "large-v2", "large-v3"]

            use_mlx_for_ui = (
                runtime == "stable-ts" and
                sys.platform == "darwin" and
                "arm" in os.uname().machine and
                load_key("whisper.stable_ts_mlx", True)
            )

            if use_mlx_for_ui:
                whisper_models = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
                                 "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo", "turbo"]
                help_text = t("Select a model compatible with MLX. 'turbo' is optimized for Apple Silicon.")

            elif runtime == 'stable-ts':
                base_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3",
                               "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"]
                custom_option = t("Custom Hugging Face Model...")
                whisper_models = base_models + [custom_option]
                help_text = t("Select a model or choose 'Custom' to enter a Hugging Face model ID.")

            # Handle model selection UI and logic
            current_model = load_key("whisper.model")

            # Determine the index for the selectbox
            if current_model in whisper_models:
                model_index = whisper_models.index(current_model)
            # If it's a custom model, it won't be in the list of options, so we select the custom option
            elif runtime == 'stable-ts' and not use_mlx_for_ui:
                 model_index = whisper_models.index(t("Custom Hugging Face Model..."))
            else:
                # Fallback for safety: if model is somehow invalid, default to the first option
                model_index = 0
                update_key("whisper.model", whisper_models[model_index])

            selected_option = st.selectbox(t("Whisper Model"), options=whisper_models,
                                           index=model_index,
                                           help=help_text)

            whisper_model_to_save = selected_option
            # If the user chose the custom option in the stable-ts non-mlx runtime
            if runtime == 'stable-ts' and not use_mlx_for_ui and selected_option == t("Custom Hugging Face Model..."):
                custom_model_id = st.text_input(
                    t("Hugging Face Model ID"),
                    value=(current_model if current_model not in base_models else "openai/whisper-large-v3")
                )
                whisper_model_to_save = custom_model_id

            if whisper_model_to_save != current_model:
                update_key("whisper.model", whisper_model_to_save)
                st.rerun()

            # Add MLX toggle for stable-ts on Apple Silicon
            if runtime == "stable-ts" and sys.platform == "darwin" and "arm" in os.uname().machine:
                use_mlx = st.toggle(
                    "🚀 " + t("Use MLX Acceleration"),
                    value=load_key("whisper.stable_ts_mlx", True), # Default to True on Apple Silicon
                    help=t("Enable MLX for faster processing on Apple Silicon. Disable if you encounter issues or want to use features not supported by MLX (like refine/regroup post-processing).")
                )
                if use_mlx != load_key("whisper.stable_ts_mlx", True):
                    update_key("whisper.stable_ts_mlx", use_mlx)
                    st.rerun()

        demucs = st.toggle(t("Use Demucs for Vocal Separation"), value=load_key("demucs"), help=t("Separate vocals from background music before transcription for better accuracy"))
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        if demucs:
            demucs_models = {
                "htdemucs": t("htdemucs (Fast, Default)"),
                "htdemucs_ft": t("htdemucs_ft (Better quality, 4x slower)")
            }
            current_demucs_model = load_key("demucs_model", "htdemucs")
            selected_demucs_model = st.selectbox(
                t("Demucs Model"),
                options=list(demucs_models.keys()),
                format_func=lambda x: demucs_models[x],
                index=list(demucs_models.keys()).index(current_demucs_model) if current_demucs_model in demucs_models else 0,
                help=t("Select the Demucs model for vocal separation.")
            )
            if selected_demucs_model != current_demucs_model:
                update_key("demucs_model", selected_demucs_model)
                st.rerun()

        burn_subtitles = st.toggle(t("Burn-in Subtitles"), value=load_key("burn_subtitles"), help=t("Whether to burn subtitles into the video, will increase processing time"))
        if burn_subtitles != load_key("burn_subtitles"):
            update_key("burn_subtitles", burn_subtitles)
            st.rerun()

        merge_subtitles = st.checkbox(
            t("Merge Subtitles"),
            value=load_key("merge_subtitles", True), # Default to True
            help=t("Check to merge short or fast subtitles for better dubbing quality.")
        )
        if merge_subtitles != load_key("merge_subtitles", True):
            update_key("merge_subtitles", merge_subtitles)
            st.rerun()
    with st.expander(t("Dubbing Settings"), expanded=True):
        configured_tts = load_key("tts_method")
        tts_methods = list_selectable_tts_methods()
        if configured_tts not in tts_methods:
            tts_methods = [configured_tts] + tts_methods
            st.warning(t("Configured TTS method is not a normal selectable route. Verify provider lifecycle before running."))
        select_tts = st.selectbox(t("TTS Method"), options=tts_methods, index=tts_methods.index(configured_tts))
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

        quality_modes = ["subtitle", "dubbing", "high_sync"]
        current_quality_mode = load_key("dubbing_quality.mode", load_key("quality_mode", "high_sync"))
        if current_quality_mode not in quality_modes:
            current_quality_mode = "high_sync"
        selected_quality_mode = st.selectbox(
            "Quality Mode",
            options=quality_modes,
            index=quality_modes.index(current_quality_mode),
            help="subtitle skips strict dubbing gates; high_sync enables duration budgets, rewrite retry, and eval."
        )
        if selected_quality_mode != current_quality_mode:
            update_key("quality_mode", selected_quality_mode)
            update_key("dubbing_quality.mode", selected_quality_mode)
            st.rerun()

        # Add rewrite toggle
        rewrite_text = st.toggle(t("Rewrite Text for Dubbing"), value=load_key("rewrite_text_for_dubbing", True), help=t("Use LLM to shorten text if it's too long for the audio slot."))
        if rewrite_text != load_key("rewrite_text_for_dubbing", True):
            update_key("rewrite_text_for_dubbing", rewrite_text)
            st.rerun()

        smoke_enabled = st.toggle("60s Smoke Run", value=load_key("smoke_test.enabled", False), help="Trim newly downloaded videos to a short sample for pipeline tuning.")
        if smoke_enabled != load_key("smoke_test.enabled", False):
            update_key("smoke_test.enabled", smoke_enabled)
            st.rerun()
        if smoke_enabled:
            smoke_seconds = st.number_input("Smoke Seconds", min_value=10, max_value=180, value=int(load_key("smoke_test.cutoff_seconds", 60)), step=10)
            if smoke_seconds != int(load_key("smoke_test.cutoff_seconds", 60)):
                update_key("smoke_test.cutoff_seconds", int(smoke_seconds))
                st.rerun()

        with st.container(border=True):
            st.caption("Dubbing Eval")
            if st.button("Run Dubbing Eval", key="run_dubbing_eval"):
                try:
                    from core.dubbing_quality import DUBBING_EVAL_JSON, write_dubbing_eval
                    summary = write_dubbing_eval()
                    st.success(f"Eval written: {summary}")
                except Exception as e:
                    st.error(f"Dubbing eval failed: {e}")
            eval_json = "output/audio/dubbing_eval.json"
            if os.path.exists(eval_json):
                try:
                    with open(eval_json, "r", encoding="utf-8") as f:
                        eval_data = json.load(f)
                    st.json(eval_data.get("summary", eval_data), expanded=False)
                except Exception as e:
                    st.caption(f"Eval summary unavailable: {e}")

        # sub settings for each tts method
        if select_tts == "f5tts":
            config_input("302ai API", "f5tts.302_api")

        elif select_tts == "mlx_router" or select_tts.startswith("mlx_"):
            st.info(t("Local MLX TTS router provides nine voice-cloning backends; experimental backends require their local runtime or API service."))
            router_backends = ["auto", "indextts2", "omnivoice", "qwen3_tts", "voxcpm2", "higgs", "dots", "zonos2", "moss", "ming"]
            forced = {
                "mlx_indextts2": "indextts2",
                "mlx_omnivoice": "omnivoice",
                "mlx_qwen3_tts": "qwen3_tts",
                "mlx_voxcpm2": "voxcpm2",
                "mlx_higgs_audio": "higgs",
                "mlx_dots_tts": "dots",
                "mlx_zonos2": "zonos2",
                "mlx_moss_tts": "moss",
                "mlx_ming_omni_tts": "ming",
            }.get(select_tts, load_key("mlx_tts.default_backend", "auto"))
            selected_backend = st.selectbox(
                "MLX TTS Backend",
                options=router_backends,
                index=router_backends.index(forced) if forced in router_backends else 0,
                help=t("Auto routes Vietnamese to IndexTTS2 and Chinese dialogue to OmniVoice by default.")
            )
            if select_tts == "mlx_router" and selected_backend != load_key("mlx_tts.default_backend", "auto"):
                update_key("mlx_tts.default_backend", selected_backend)
                st.rerun()
            if st.button("Check MLX TTS Backends", key="check_mlx_tts"):
                try:
                    from core.providers.mlx_tts import list_backend_status
                    st.json(list_backend_status(), expanded=False)
                except Exception as e:
                    st.error(f"MLX TTS check failed: {e}")

def check_api():
    try:
        resp = ask_gpt("This is a test, response 'message':'success' in json format.",
                      response_json=True, log_title='None')
        return resp.get('message') == 'success'
    except Exception:
        return False
