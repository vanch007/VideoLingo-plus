# use try-except to avoid error when installing
try:
    from . import config_utils
    from . import timing_utils
    from . import step1_ytdlp
    from . import step2_whisperX
    from . import step3_1_spacy_split
    from . import step3_2_splitbymeaning
    from . import step4_1_summarize
    from . import step4_2_translate_all
    from . import step5_splitforsub
    from . import step6_generate_final_timeline
    from . import step7_merge_sub_to_vid
    from . import step8_1_gen_audio_task
    from . import step8_2_gen_dub_chunks
    from . import step9_extract_refer_audio
    from . import step10_gen_audio
    from . import step11_merge_full_audio
    from . import step12_merge_dub_to_vid
    from . import translate_once
    from . import ask_gpt
    from . import prompts_storage
    from . import delete_retry_dubbing
    from . import onekeycleanup
    from . import pypi_autochoose
except ImportError as e:
    print(f"Warning: Some modules could not be imported during installation: {e}")