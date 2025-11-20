
import streamlit as st
import pandas as pd
from core.timing_utils import get_formatted_timings, clear_timings

def display_timing_statistics_component(key_suffix="main"):
    """
    Display processing time statistics with an improved UI.
    """
    try:
        # Check if we're being called during cleanup
        import traceback
        stack = traceback.extract_stack()
        caller_files = [frame[0] for frame in stack]
        in_cleanup = any('onekeycleanup.py' in file for file in caller_files)

        if in_cleanup:
            return

        timings = get_formatted_timings()
        
        if not timings:
            st.info("暂无耗时数据，运行处理后将显示在这里")
            return

        # Categorize data
        categories = {
            "project": [],
            "download": [],
            "text": [],
            "audio": [],
            "other": []
        }
        
        text_steps = ['转录', 'NLP分句', 'LLM分句', '摘要', '翻译', '字幕分割', '时间轴对齐', '字幕合并到视频', '整体字幕处理']
        audio_steps = ['生成配音任务', '生成配音分块', '提取参考音频', '生成配音', '合并配音', '配音合并到视频', '整体配音处理']
        
        for item in timings:
            step = item['步骤']
            if step in ['项目开始时间', '项目总耗时']:
                categories['project'].append(item)
            elif step in ['下载视频', '上传视频']:
                categories['download'].append(item)
            elif step in text_steps:
                categories['text'].append(item)
            elif step in audio_steps:
                categories['audio'].append(item)
            else:
                categories['other'].append(item)

        # Main Container
        with st.container(border=True):
            st.markdown("### ⏱️ 处理耗时统计")
            
            # Calculate totals for text and audio
            text_total_seconds = sum(_parse_time_str(item['耗时']) for item in categories['text'] if item['步骤'] == '整体字幕处理')
            audio_total_seconds = sum(_parse_time_str(item['耗时']) for item in categories['audio'] if item['步骤'] == '整体配音处理')
            
            # If specific total steps are missing, sum up all steps (approximate)
            if text_total_seconds == 0 and categories['text']:
                 text_total_seconds = sum(_parse_time_str(item['耗时']) for item in categories['text'] if item['步骤'] != '整体字幕处理')
            if audio_total_seconds == 0 and categories['audio']:
                 audio_total_seconds = sum(_parse_time_str(item['耗时']) for item in categories['audio'] if item['步骤'] != '整体配音处理')

            # Top Metrics (Project Total Time)
            total_time_str = next((item['耗时'] for item in categories['project'] if item['步骤'] == '项目总耗时'), None)
            start_time = next((item['耗时'] for item in categories['project'] if item['步骤'] == '项目开始时间'), None)
            
            # Fallback for total time if missing or too small (< 1s) but we have stage times
            if (not total_time_str or _parse_time_str(total_time_str) < 1) and (text_total_seconds > 1 or audio_total_seconds > 1):
                calculated_total = text_total_seconds + audio_total_seconds
                total_time_str = _format_seconds(calculated_total)

            cols = st.columns(2)
            if total_time_str:
                cols[0].metric("项目总耗时", total_time_str)
            if start_time:
                cols[1].metric("开始时间", start_time)
                
            st.divider()

            # Tabs for detailed breakdown
            tabs = st.tabs(["📊 概览", "📝 字幕处理", "🎵 配音处理"])
            
            # Tab 1: Overview (Download/Upload + Other + Stage Summaries)
            with tabs[0]:
                # Stage Summary Table
                summary_data = []
                if text_total_seconds > 0:
                    summary_data.append({"阶段": "📝 字幕处理", "耗时": _format_seconds(text_total_seconds)})
                if audio_total_seconds > 0:
                    summary_data.append({"阶段": "🎵 配音处理", "耗时": _format_seconds(audio_total_seconds)})
                
                if summary_data:
                    st.markdown("#### 📈 阶段汇总")
                    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

                if categories['download']:
                    st.markdown("#### 📥 下载/上传")
                    df_download = pd.DataFrame(categories['download'])
                    st.dataframe(df_download, hide_index=True, use_container_width=True)
                
                if categories['other']:
                    st.markdown("#### 📋 其他")
                    df_other = pd.DataFrame(categories['other'])
                    st.dataframe(df_other, hide_index=True, use_container_width=True)
                    
                if not summary_data and not categories['download'] and not categories['other']:
                    st.info("暂无概览数据")

            # Tab 2: Text Processing
            with tabs[1]:
                if categories['text']:
                    # Sort text steps
                    text_step_order = {step: i for i, step in enumerate(text_steps)}
                    categories['text'].sort(key=lambda x: text_step_order.get(x['步骤'], 100))
                    
                    df_text = pd.DataFrame(categories['text'])
                    st.dataframe(df_text, hide_index=True, use_container_width=True)
                    
                    # Chart
                    try:
                        chart_data = _prepare_chart_data(categories['text'])
                        if not chart_data.empty:
                            st.bar_chart(chart_data.set_index('步骤'), color="#4B8BF5")
                    except Exception:
                        pass
                else:
                    st.info("暂无字幕处理数据")

            # Tab 3: Audio Processing
            with tabs[2]:
                if categories['audio']:
                    # Sort audio steps
                    audio_step_order = {step: i for i, step in enumerate(audio_steps)}
                    categories['audio'].sort(key=lambda x: audio_step_order.get(x['步骤'], 100))
                    
                    df_audio = pd.DataFrame(categories['audio'])
                    st.dataframe(df_audio, hide_index=True, use_container_width=True)
                    
                    # Chart
                    try:
                        chart_data = _prepare_chart_data(categories['audio'])
                        if not chart_data.empty:
                            st.bar_chart(chart_data.set_index('步骤'), color="#FF4B4B")
                    except Exception:
                        pass
                else:
                    st.info("暂无配音处理数据")

            # Clear Button
            if st.button("🗑️ 清除统计", key=f"clear_timing_{key_suffix}"):
                clear_timings()
                st.rerun()

    except Exception as e:
        st.error(f"显示耗时统计出错: {str(e)}")

def _parse_time_str(time_str):
    """Parse time string to seconds."""
    try:
        seconds = 0
        if '小时' in time_str:
            parts = time_str.replace('小时', ':').replace('分', ':').replace('秒', '').split(':')
            seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif '分' in time_str:
            parts = time_str.replace('分', ':').replace('秒', '').split(':')
            seconds = float(parts[0]) * 60 + float(parts[1])
        elif '秒' in time_str:
            seconds = float(time_str.replace('秒', ''))
        return seconds
    except:
        return 0

def _format_seconds(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} 分 {seconds:.2f} 秒"
    else:
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        seconds %= 60
        return f"{hours} 小时 {minutes} 分 {seconds:.2f} 秒"

def _prepare_chart_data(items):
    data = []
    for item in items:
        seconds = _parse_time_str(item['耗时'])
        if seconds > 0:
            data.append({'步骤': item['步骤'], '耗时(秒)': seconds})
            
    return pd.DataFrame(data)
