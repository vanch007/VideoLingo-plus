import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from st_components.imports_and_utils import ask_gpt
import streamlit as st
from core.config_utils import update_key, load_key
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

def page_setting():

    display_language = st.selectbox("Display Language 🌐",
                                  options=list(DISPLAY_LANGUAGES.keys()),
                                  index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")))
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key")
        config_input(t("BASE_URL"), "api.base_url", help=t("Openai format, will add /v1/chat/completions automatically"))

        c1, c2 = st.columns([4, 1])
        with c1:
            config_input(t("MODEL"), "api.model", help=t("click to check API validity")+ " 👉")
        with c2:
            if st.button("📡", key="api"):
                st.toast(t("API Key is valid") if check_api() else t("API Key is invalid"),
                        icon="✅" if check_api() else "❌")

    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇸🇦 العربية": "ar",
                "🇦🇲 Հայերեն": "hy",
                "🇦🇿 Azərbaycan": "az",
                "🇧🇾 Беларуская": "be",
                "🇧🇦 Bosanski": "bs",
                "🇧🇬 Български": "bg",
                "🇲🇲 မြန်မာ": "my",
                "🇪🇸 Català": "ca",
                "🇨🇳 简体中文": "zh",
                "🇭🇷 Hrvatski": "hr",
                "🇨🇿 Čeština": "cs",
                "🇩🇰 Dansk": "da",
                "🇳🇱 Nederlands": "nl",
                "🇺🇸 English": "en",
                "🇪🇪 Eesti": "et",
                "🇫🇮 Suomi": "fi",
                "🇫🇷 Français": "fr",
                "🇪🇸 Galego": "gl",
                "🇩🇪 Deutsch": "de",
                "🇬🇷 Ελληνικά": "el",
                "🇮🇳 ગુજરાતી": "gu",
                "🇮🇱 עברית": "he",
                "🇮🇳 हिन्दी": "hi",
                "🇭🇺 Magyar": "hu",
                "🇮🇸 Íslenska": "is",
                "🇮🇩 Bahasa Indonesia": "id",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja",
                "🇮🇳 ಕನ್ನಡ": "kn",
                "🇰🇿 Қазақша": "kk",
                "🇰🇭 ខ្មែរ": "km",
                "🇰🇷 한국어": "ko",
                "🇱🇦 ລາວ": "lo",
                "🇱🇻 Latviešu": "lv",
                "🇱🇹 Lietuvių": "lt",
                "🇲🇰 Македонски": "mk",
                "🇲🇾 Bahasa Melayu": "ms",
                "🇮🇳 मराठी": "mr",
                "🇲🇳 Монгол": "mn",
                "🇳🇵 नेपाली": "ne",
                "🇳🇴 Norsk": "no",
                "🇮🇷 فارسی": "fa",
                "🇵🇱 Polski": "pl",
                "🇵🇹 Português": "pt",
                "🇷🇴 Română": "ro",
                "🇷🇺 Русский": "ru",
                "🇷🇸 Српски": "sr",
                "🇸🇰 Slovenčina": "sk",
                "🇸🇮 Slovenščina": "sl",
                "🇪🇸 Español": "es",
                "🇸🇪 Svenska": "sv",
                "🇮🇳 தமிழ்": "ta",
                "🇮🇳 తెలుగు": "te",
                "🇹🇭 ภาษาไทย": "th",
                "🇹🇷 Türkçe": "tr",
                "🇺🇦 Українська": "uk",
                "🇵🇰 اردو": "ur",
                "🇻🇳 Tiếng Việt": "vi",
                "🇬🇧 Cymraeg": "cy",
            }
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language"))
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        # add runtime selection in v2.2.0
        runtime = st.selectbox(t("WhisperX Runtime"), options=["local", "stable-ts", "cloud"], index=["local", "stable-ts", "cloud"].index(load_key("whisper.runtime")), help=t("Local runtime requires >8GB GPU, stable-ts provides better timestamps, cloud runtime requires 302ai API key"))
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()

        if runtime == "local":
            config_input(t("HuggingFace Token"), "hf_token", help=t("Required for Diarization (Speaker Identification). Get it from https://huggingface.co/settings/tokens"))
            
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

        if runtime == "cloud":
            config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")

        with c2:
            target_language = st.text_input(t("Target Lang"), value=load_key("target_language"), help=t("Input any language in natural language, as long as llm can understand"))
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        demucs = st.toggle(t("Vocal separation enhance"), value=load_key("demucs"), help=t("Recommended for videos with loud background noise, but will increase processing time"))
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
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
        tts_methods = ["azure_tts", "openai_tts", "fish_tts", "sf_fish_tts", "edge_tts", "gpt_sovits", "custom_tts", "sf_indextts2", "f5tts", "index_tts2"]
        select_tts = st.selectbox(t("TTS Method"), options=tts_methods, index=tts_methods.index(load_key("tts_method")))
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()
            
        # Add rewrite toggle
        rewrite_text = st.toggle(t("Rewrite Text for Dubbing"), value=load_key("rewrite_text_for_dubbing", True), help=t("Use LLM to shorten text if it's too long for the audio slot."))
        if rewrite_text != load_key("rewrite_text_for_dubbing", True):
            update_key("rewrite_text_for_dubbing", rewrite_text)
            st.rerun()

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":
            config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")

            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic")
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode")) if load_key("sf_fish_tts.mode") in mode_options.keys() else 0
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input("Voice", "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input("302ai API", "openai_tts.api_key")
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_tts":
            config_input("302ai API", "fish_tts.api_key")
            fish_tts_character = st.selectbox(t("Fish TTS Character"), options=list(load_key("fish_tts.character_id_dict").keys()), index=list(load_key("fish_tts.character_id_dict").keys()).index(load_key("fish_tts.character")))
            if fish_tts_character != load_key("fish_tts.character"):
                update_key("fish_tts.character", fish_tts_character)
                st.rerun()

        elif select_tts == "azure_tts":
            config_input("302ai API", "azure_tts.api_key")
            config_input(t("Azure Voice"), "azure_tts.voice")

        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")

            refer_mode_options = {1: t("Mode 1: Use provided reference audio only"), 2: t("Mode 2: Use first audio from video as reference"), 3: t("Mode 3: Use each audio from video as reference")}
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(load_key("gpt_sovits.refer_mode")),
                help=t("Configure reference audio mode for GPT-SoVITS")
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()

        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "sf_indextts2":
            config_input(t("SiliconFlow API Key"), "sf_indextts2.api_key")
            config_number_input(t("Max Workers"), "max_workers", help=t("Number of parallel processes for TTS generation."), min_value=1, step=1)

            # 获取固定声音列表
            voice_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "all_tts_functions", "voice")
            if os.path.exists(voice_dir):
                # 过滤掉 .DS_Store 等隐藏文件
                fixed_voices = [d for d in os.listdir(voice_dir) if os.path.isdir(os.path.join(voice_dir, d)) and not d.startswith('.')]
            else:
                fixed_voices = []

            # 添加克隆模式选择
            clone_mode_options = {
                "dynamic": t("Dynamic Clone"),
                "fixed": t("Fixed Clone")
            }
            
            try:
                default_clone_mode = load_key("sf_indextts2.clone_mode")
            except KeyError:
                default_clone_mode = "dynamic"

            selected_clone_mode = st.radio(
                t("Clone Mode"),
                options=list(clone_mode_options.keys()),
                format_func=lambda x: clone_mode_options[x],
                index=list(clone_mode_options.keys()).index(default_clone_mode),
                key="cosyvoice_clone_mode"
            )
            if selected_clone_mode != default_clone_mode:
                update_key("sf_indextts2.clone_mode", selected_clone_mode)
                st.rerun()

            if selected_clone_mode == "fixed":
                if fixed_voices:
                    try:
                        default_fixed_voice = load_key("sf_indextts2.fixed_voice")
                        if default_fixed_voice not in fixed_voices:
                            default_fixed_voice = fixed_voices[0]
                            update_key("sf_indextts2.fixed_voice", default_fixed_voice)
                    except KeyError:
                        default_fixed_voice = fixed_voices[0] if fixed_voices else None
                        if default_fixed_voice:
                            update_key("sf_indextts2.fixed_voice", default_fixed_voice)

                    if default_fixed_voice:
                        selected_fixed_voice = st.selectbox(
                            t("Fixed Voice"),
                            options=fixed_voices,
                            index=fixed_voices.index(default_fixed_voice),
                            key="cosyvoice_fixed_voice"
                        )
                        if selected_fixed_voice != default_fixed_voice:
                            update_key("sf_indextts2.fixed_voice", selected_fixed_voice)
                            st.rerun()
                else:
                    st.warning(t("No fixed voices found. Please add voices in 'core/all_tts_functions/voice/' directory."))

        elif select_tts == "f5tts":
            config_input("302ai API", "f5tts.302_api")

def check_api():
    try:
        resp = ask_gpt("This is a test, response 'message':'success' in json format.",
                      response_json=True, log_title='None')
        return resp.get('message') == 'success'
    except Exception:
        return False