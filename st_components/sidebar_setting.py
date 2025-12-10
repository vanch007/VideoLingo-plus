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
        asr_options = {"WhisperX": "local", "stable-ts": "stable-ts", "FunASR": "funasr"}
        asr_display = list(asr_options.keys())
        asr_values = list(asr_options.values())
        current_runtime = load_key("whisper.runtime") if load_key("whisper.runtime") in asr_values else "local"
        current_index = asr_values.index(current_runtime)
        
        selected_asr = st.selectbox(
            t("ASR Model"),
            options=asr_display,
            index=current_index,
            help=t("WhisperX requires >8GB GPU, stable-ts provides better timestamps, FunASR supports Chinese with speaker diarization")
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

        if runtime == "funasr":
            # FunASR model selection
            funasr_models = {"SenseVoice": "sensevoice", "Paraformer": "paraformer"}
            funasr_display = list(funasr_models.keys())
            funasr_values = list(funasr_models.values())
            current_funasr_model = load_key("funasr.model", "sensevoice")
            current_funasr_index = funasr_values.index(current_funasr_model) if current_funasr_model in funasr_values else 0
            
            selected_funasr_model = st.selectbox(
                t("FunASR Model"),
                options=funasr_display,
                index=current_funasr_index,
                help=t("SenseVoice: newer multi-functional model. Paraformer: classic stable model with speaker diarization.")
            )
            if funasr_models[selected_funasr_model] != current_funasr_model:
                update_key("funasr.model", funasr_models[selected_funasr_model])
                st.rerun()
            
            # Speaker diarization toggle (only for Paraformer)
            if funasr_models[selected_funasr_model] == "paraformer":
                enable_spk = st.toggle(
                    t("Enable Speaker Diarization"),
                    value=load_key("funasr.enable_spk", True),
                    help=t("Use CAM++ for speaker identification. Requires more memory.")
                )
                if enable_spk != load_key("funasr.enable_spk", True):
                    update_key("funasr.enable_spk", enable_spk)
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
        tts_methods = ["edge_tts", "gpt_sovits", "custom_tts", "sf_indextts2", "index_tts2", "piper_tts", "indonesian_tts", "voxcpm_tts"]
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
        if select_tts == "gpt_sovits":
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

        elif select_tts == "piper_tts":
            st.info(t("Piper TTS is a fast local TTS engine. First run will download voice models."))
            # Use load_key with default for new config keys
            current_voice = load_key("piper_tts.voice", "en_US-lessac-medium")
            new_voice = st.text_input(t("Piper Voice"), value=current_voice, help=t("Voice name, e.g. en_US-lessac-medium, zh_CN-huayan-medium"))
            if new_voice != current_voice:
                update_key("piper_tts.voice", new_voice)
                st.rerun()

        elif select_tts == "indonesian_tts":
            st.info(t("Indonesian TTS uses native Indonesian VITS model with 83 speakers (Indonesian, Javanese, Sundanese)."))
            
            # Load speaker list from model
            speakers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "indonesian_tts", "speakers.pth")
            indonesian_speakers = []
            
            if os.path.exists(speakers_path):
                import torch
                try:
                    speakers_dict = torch.load(speakers_path, weights_only=False)
                    indonesian_speakers = sorted(speakers_dict.keys())
                except Exception as e:
                    st.warning(f"Could not load speakers: {e}")
            
            if indonesian_speakers:
                # Group speakers by category
                id_speakers = [s for s in indonesian_speakers if s in ["wibowo", "ardi", "gadis"]]
                jv_speakers = [s for s in indonesian_speakers if s.startswith("JV-")]
                su_speakers = [s for s in indonesian_speakers if s.startswith("SU-")]
                
                # Display speaker info
                st.caption(f"📢 {t('Available speakers')}: {len(id_speakers)} Indonesian, {len(jv_speakers)} Javanese, {len(su_speakers)} Sundanese")
                
                # Speaker selection dropdown - prioritize Indonesian speakers
                all_speakers = id_speakers + jv_speakers + su_speakers
                
                default_speaker = load_key("indonesian_tts.speaker", "wibowo")
                if default_speaker not in all_speakers:
                    default_speaker = "wibowo" if "wibowo" in all_speakers else all_speakers[0]
                    update_key("indonesian_tts.speaker", default_speaker)
                
                selected_speaker = st.selectbox(
                    t("Indonesian TTS Speaker"),
                    options=all_speakers,
                    index=all_speakers.index(default_speaker),
                    key="indonesian_tts_speaker",
                    help=t("Select a speaker voice. Indonesian voices: wibowo, ardi, gadis. JV = Javanese, SU = Sundanese.")
                )
                if selected_speaker != default_speaker:
                    update_key("indonesian_tts.speaker", selected_speaker)
                    st.rerun()
            else:
                st.warning(t("Indonesian TTS model not found. Please download from https://github.com/Wikidepia/indonesian-tts/releases/tag/v1.2"))

        elif select_tts == "voxcpm_tts":
            st.info(t("VoxCPM TTS is a tokenizer-free speech synthesis with realistic voice cloning. Requires local VoxCPM service running."))
            
            # API URL configuration
            current_api_url = load_key("voxcpm_tts.api_url", "http://127.0.0.1:7860")
            new_api_url = st.text_input(
                t("VoxCPM API URL"),
                value=current_api_url,
                help=t("Local VoxCPM Gradio server URL, default: http://127.0.0.1:7860")
            )
            if new_api_url != current_api_url:
                update_key("voxcpm_tts.api_url", new_api_url)
                st.rerun()
            
            # Prompt enhancement toggle
            use_enhancement = st.toggle(
                t("Prompt Speech Enhancement"),
                value=load_key("voxcpm_tts.use_prompt_enhancement", False),
                help=t("Enable for clearer voice (16kHz). Disable for higher quality cloning (up to 44.1kHz).")
            )
            if use_enhancement != load_key("voxcpm_tts.use_prompt_enhancement", False):
                update_key("voxcpm_tts.use_prompt_enhancement", use_enhancement)
                st.rerun()
            
            # Text normalization toggle
            normalize = st.toggle(
                t("Text Normalization"),
                value=load_key("voxcpm_tts.normalize", False),
                help=t("Enable for regular text (handles numbers, abbreviations). Disable for phoneme input.")
            )
            if normalize != load_key("voxcpm_tts.normalize", False):
                update_key("voxcpm_tts.normalize", normalize)
                st.rerun()
            
            # Output denoising toggle
            denoise = st.toggle(
                t("Output Denoising"),
                value=load_key("voxcpm_tts.denoise", False),
                help=t("Enable external denoising (may cause distortion, limits sample rate to 16kHz).")
            )
            if denoise != load_key("voxcpm_tts.denoise", False):
                update_key("voxcpm_tts.denoise", denoise)
                st.rerun()

def check_api():
    try:
        resp = ask_gpt("This is a test, response 'message':'success' in json format.",
                      response_json=True, log_title='None')
        return resp.get('message') == 'success'
    except Exception:
        return False