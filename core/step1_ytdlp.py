import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import glob
import re
import shutil
import subprocess
from core.config_utils import load_key

def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def _browser_cookie_tuple(value):
    if not value:
        return None
    browser, _, profile = value.partition(":")
    return (browser, profile or None, None, None)

def extract_clean_url(text: str) -> str:
    """Extract a clean HTTP/HTTPS URL from user input, removing mobile share text."""
    if not text:
        return ""
    text_str = str(text).strip()
    match = re.search(r'https?://[^\s<>\"\'\)]+', text_str)
    if match:
        return match.group(0).rstrip('.,;!?')
    return text_str

def resolve_redirect_url(url: str, timeout: int = 10) -> str:
    """Follow HTTP redirects to uncover destination canonical URL (useful for shortlinks)."""
    try:
        import requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/16.0 Mobile/15E148 Safari/604.1"
            )
        }
        resp = requests.get(url, headers=headers, allow_redirects=True, timeout=timeout)
        if resp.url:
            return resp.url
    except Exception:
        pass
    return url

def download_douyin_watermark_free(url: str, save_path: str = 'output') -> str | None:
    """Download watermark-free video directly from Douyin using mobile/aweme endpoints as fallback."""
    import requests
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 8.0.0; SM-G955U Build/R16NW) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/116.0.0.0 Mobile Safari/537.36"
        ),
        "Referer": "https://www.douyin.com/?is_from_mobile_home=1&recommend=1",
    }
    real_url = resolve_redirect_url(url)
    video_id = None
    m = re.search(r"/(?:video|note)/([0-9]+)", real_url)
    if m:
        video_id = m.group(1)
    if not video_id:
        for param in ["modal_id", "note_id", "item_id", "video_id"]:
            m = re.search(rf"{param}=([0-9]+)", real_url)
            if m:
                video_id = m.group(1)
                break
    if not video_id:
        m = re.search(r"/([0-9]{15,})", real_url)
        if m:
            video_id = m.group(1)

    if not video_id:
        return None

    video_url = None
    title = f"douyin_{video_id}"
    for share_type in ("video", "note"):
        try:
            share_page = f"https://www.iesdouyin.com/share/{share_type}/{video_id}/"
            r = requests.get(share_page, headers=headers, timeout=12)
            if r.status_code == 200:
                m_json = re.search(r"window\._ROUTER_DATA\s*=\s*(.*?)</script>", r.text, re.DOTALL)
                if m_json:
                    data = json.loads(m_json.group(1).strip())
                    loader = data.get("loaderData", {})
                    for _, v in loader.items():
                        if isinstance(v, dict):
                            for _, v2 in v.items():
                                if isinstance(v2, dict) and "item_list" in v2 and v2["item_list"]:
                                    item = v2["item_list"][0]
                                    desc = item.get("desc")
                                    if desc:
                                        title = sanitize_filename(desc)
                                    uri = item.get("video", {}).get("play_addr", {}).get("uri")
                                    if uri:
                                        video_url = f"https://www.douyin.com/aweme/v1/play/?video_id={uri}"
                                        break
                        if video_url:
                            break
        except Exception:
            pass
        if video_url:
            break

    if not video_url:
        try:
            api_url = (
                f"https://www.douyin.com/aweme/v1/web/aweme/detail/?aweme_id={video_id}"
                "&device_platform=webapp&aid=6383&channel=channel_pc_web"
            )
            r = requests.get(api_url, headers={
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
                "Referer": "https://www.douyin.com/",
            }, timeout=12)
            if r.status_code == 200:
                data = r.json()
                ad = data.get("aweme_detail", {})
                if ad:
                    desc = ad.get("desc")
                    if desc:
                        title = sanitize_filename(desc)
                    url_list = ad.get("video", {}).get("play_addr", {}).get("url_list", [])
                    if url_list:
                        video_url = next((u for u in url_list if "douyinvod.com" in u), url_list[0])
        except Exception:
            pass

    if not video_url:
        return None

    os.makedirs(save_path, exist_ok=True)
    target_file = os.path.join(save_path, f"{title}.mp4")
    try:
        resp = requests.get(video_url, headers=headers, stream=True, timeout=30)
        if resp.status_code == 403:
            h2 = headers.copy()
            h2.pop("Referer", None)
            resp = requests.get(video_url, headers=h2, stream=True, timeout=30)
        resp.raise_for_status()
        with open(target_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if os.path.isfile(target_file) and os.path.getsize(target_file) > 1024:
            return target_file
    except Exception:
        if os.path.isfile(target_file):
            try:
                os.remove(target_file)
            except OSError:
                pass
        raise
    return None

def _patch_bilibili_extractor():
    """Ensure Bilibili overseas requests prefer reliable CDN mirrors (like bilivideo.com) over Akamai."""
    try:
        from yt_dlp.extractor.bilibili import BiliBiliIE
        if getattr(BiliBiliIE, "_backup_url_patched", False):
            return
        orig_extract = BiliBiliIE.extract_formats

        def extract_formats_patched(self, play_info):
            dash = play_info.get("dash", {})
            for kind in ("audio", "video"):
                for item in dash.get(kind, []):
                    base = item.get("baseUrl") or item.get("base_url") or ""
                    if "akamaized.net" in base:
                        backups = item.get("backupUrl") or item.get("backup_url") or []
                        good_backup = next((u for u in backups if "bilivideo.com" in u), None)
                        if good_backup:
                            item["baseUrl"] = good_backup
                            item["base_url"] = good_backup
            return orig_extract(self, play_info)

        BiliBiliIE.extract_formats = extract_formats_patched
        BiliBiliIE._backup_url_patched = True
    except Exception:
        pass

def download_video_ytdlp(url, save_path='output', resolution='1080', cutoff_time=None, cookies_from_browser=None):
    _patch_bilibili_extractor()
    clean_url = extract_clean_url(url)
    if not clean_url:
        raise ValueError("Invalid video URL provided")

    # Resolve shortlinks (e.g. v.douyin.com) to canonical URLs for extractor matching
    if "v.douyin.com" in clean_url:
        resolved = resolve_redirect_url(clean_url)
        if resolved:
            clean_url = resolved

    res_str = str(resolution).lower().strip().replace('p', '')
    if res_str in ('best', 'auto', 'original', 'max'):
        format_spec = 'bestvideo+bestaudio/best'
    elif res_str.isdigit():
        height = int(res_str)
        format_spec = (
            f'bestvideo[height<={height}]+bestaudio/'
            f'bestvideo[height<={height}]+best/'
            f'best[height<={height}]/'
            f'bestvideo+bestaudio/best'
        )
    else:
        format_spec = (
            'bestvideo[height<=1080]+bestaudio/'
            'bestvideo[height<=1080]+best/'
            'best[height<=1080]/'
            'bestvideo+bestaudio/best'
        )

    os.makedirs(save_path, exist_ok=True)

    http_headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    }
    if 'bilibili.com' in clean_url or 'b23.tv' in clean_url:
        http_headers['Referer'] = 'https://www.bilibili.com/'

    ydl_opts = {
        'format': format_spec,
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'writethumbnail': True,
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegThumbnailsConvertor',
            'format': 'jpg',
        }],
        'retries': 5,
        'fragment_retries': 5,
        'http_headers': http_headers,
    }
    if shutil.which("node"):
        ydl_opts['js_runtimes'] = {'node': {}}
        ydl_opts['remote_components'] = ['ejs:github']

    if not cookies_from_browser:
        cookies_from_browser = os.environ.get("VIDEOLINGO_YTDLP_COOKIES_FROM_BROWSER")
    if not cookies_from_browser:
        try:
            cookies_from_browser = load_key("cookies_from_browser", default="") or None
        except Exception:
            cookies_from_browser = None
    if not cookies_from_browser:
        chrome_cookie_path = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/Cookies")
        if os.path.exists(chrome_cookie_path):
            cookies_from_browser = "chrome"
    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = _browser_cookie_tuple(cookies_from_browser)

    # Dependency maintenance is separate from a media job. The environment
    # checked at startup must remain the environment used by every stage.
    from yt_dlp import YoutubeDL
    ytdlp_succeeded = False
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])
        ytdlp_succeeded = True
    except Exception as e:
        message = str(e)
        if 'cookiesfrombrowser' in ydl_opts and "cookies" in message.lower() and (
                "could not find" in message or "Failed to decrypt" in message or "Extracting cookies" in message):
            # A configured browser without a usable cookie store must not block public media.
            print(f"⚠️ Browser cookies unavailable ({message.splitlines()[0][:120]}); retrying without cookies...")
            ydl_opts_plain = {k: v for k, v in ydl_opts.items() if k != 'cookiesfrombrowser'}
            try:
                with YoutubeDL(ydl_opts_plain) as ydl_plain:
                    ydl_plain.download([clean_url])
                ytdlp_succeeded = True
                message = ""
            except Exception as plain_err:
                e, message = plain_err, str(plain_err)
        if message and ("Sign in to confirm" in message or "not a bot" in message or "Sign in" in message):
            print("⚠️ YouTube anti-bot triggered; scanning all local Chrome profiles for valid cookies...")
            import glob
            base_dir = os.path.expanduser("~/Library/Application Support/Google/Chrome")
            candidate_profiles = ["Profile 25", "Profile 1", "Profile 9", "Profile 4", "Profile 12", "Default"]
            # append other detected profiles
            for c_path in glob.glob(f"{base_dir}/**/Cookies", recursive=True):
                p_rel = os.path.relpath(c_path, base_dir).split(os.sep)[0]
                if p_rel not in candidate_profiles and (p_rel == "Default" or p_rel.startswith("Profile ")):
                    candidate_profiles.append(p_rel)
            for prof in candidate_profiles:
                try:
                    print(f"  🔄 Trying Chrome profile: {prof}...")
                    ydl_opts_retry = dict(ydl_opts)
                    ydl_opts_retry['cookiesfrombrowser'] = ('chrome', prof, None, None)
                    with YoutubeDL(ydl_opts_retry) as ydl_retry:
                        ydl_retry.download([clean_url])
                    ytdlp_succeeded = True
                    print(f"  ✅ Successfully downloaded using Chrome profile: {prof}")
                    break
                except Exception as retry_err:
                    pass
            if not ytdlp_succeeded:
                raise RuntimeError(
                    "YouTube requires authenticated cookies for this request. "
                    "All local Chrome profiles attempted failed. Please open YouTube in Chrome to refresh login."
                ) from e
        if any(d in clean_url for d in ("douyin.com", "iesdouyin.com")):
            try:
                print("🔄 yt-dlp 解析抖音未完成，正在切换至原生无水印直连通道...")
                fallback_file = download_douyin_watermark_free(clean_url, save_path=save_path)
                if fallback_file and os.path.isfile(fallback_file):
                    ytdlp_succeeded = True
            except Exception as fallback_err:
                raise RuntimeError(f"抖音无水印视频下载失败: yt-dlp ({e}), 原生提取 ({fallback_err})") from e
        if not ytdlp_succeeded:
            raise
    
    # Check and rename files after download
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))

    # cut the video to make demo
    if cutoff_time:
        print(f"Cutoff time: {cutoff_time}, Now checking video duration...")
        video_file = find_video_files(save_path)
        
        # Use librosa to get video duration
        import librosa
        duration = librosa.get_duration(filename=video_file)
        
        if duration > cutoff_time:
            print(f"Video duration ({duration:.2f}s) is longer than cutoff time. Cutting the video...")
            file_name, file_extension = os.path.splitext(video_file)
            trimmed_file = f"{file_name}_trim{file_extension}"
            ffmpeg_cmd = ['ffmpeg', '-i', video_file, '-t', str(cutoff_time), '-c', 'copy', trimmed_file]
            print("🎬 Start cutting video...")
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8')
            for line in process.stdout:
                print(line, end='')
            process.wait()
            print(f"✅ Video has been cut to the first {cutoff_time} seconds")
            
            # Remove the original file and rename the trimmed file
            os.remove(video_file)
            os.rename(trimmed_file, video_file)
            print(f"Original file removed and trimmed file renamed to {os.path.basename(video_file)}")
        else:
            print(f"Video duration ({duration:.2f}s) is not longer than cutoff time. No need to cut.")

    # Ensure canonical source.mp4 exists
    ensure_canonical_source_video(save_path)


def ensure_canonical_source_video(save_path="output") -> str:
    """Ensure canonical output/source.mp4 exists alongside any named download/upload."""
    canonical_target = os.path.join(save_path, "source.mp4")
    if os.path.isfile(canonical_target) and os.path.getsize(canonical_target) > 0:
        return canonical_target
    allowed = load_key("allowed_video_formats", ["mp4", "mkv", "webm", "avi", "mov"])
    v_files = [f for f in glob.glob(save_path + "/*") if os.path.splitext(f)[1][1:].lower() in allowed]
    named_files = [f for f in v_files
                   if not os.path.basename(f).startswith(("AI配音", "AI字幕", "AI_", "output", "_", "dub", "preview"))]
    if not named_files:
        raise ValueError(f"No video files found in {save_path} to create canonical source.mp4")
    if len(named_files) != 1:
        raise ValueError('Ambiguous source video; canonical import requires exactly one original')
    src_file = named_files[0]
    orig_name = os.path.splitext(os.path.basename(src_file))[0]
    title_meta_path = os.path.join(save_path, "video_title.json")
    try:
        with open(title_meta_path, "w", encoding="utf-8") as f:
            json.dump({"title": orig_name}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if src_file.lower().endswith(".mp4"):
        print(f"🎬 Moving {src_file} -> {canonical_target} (keep single video copy)...")
        shutil.move(src_file, canonical_target)
    else:
        print(f"🎬 Transcoding {src_file} -> {canonical_target} and removing original...")
        remux_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", src_file, "-map", "0", "-c", "copy", canonical_target]
        res = subprocess.run(remux_cmd)
        if res.returncode != 0 or not os.path.isfile(canonical_target) or os.path.getsize(canonical_target) == 0:
            trans_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", src_file, "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", canonical_target]
            subprocess.run(trans_cmd, check=True)
        try:
            os.remove(src_file)
        except OSError:
            pass
    print(f"✅ Canonical source video created: {canonical_target}")
    return canonical_target

def find_video_files(save_path='output'):
    video_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")]
    # change \ to /, this happen on windows
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    canonical = [file for file in video_files if os.path.splitext(os.path.basename(file))[0] == "source"]
    canonical_mp4 = [file for file in canonical if os.path.basename(file) == 'source.mp4']
    if canonical_mp4:
        return canonical_mp4[0]
    if len(canonical) == 1:
        return canonical[0]
    video_files = [file for file in video_files if not os.path.basename(file).startswith(("AI配音", "AI字幕", "AI_", "output", "_", "dub", "preview"))]
    if len(video_files) == 0:
        raise ValueError(f"No video files found in {save_path}")
    if len(video_files) != 1:
        raise ValueError("Ambiguous source video; import one canonical output/source.mp4 before continuing")
    return video_files[0]

if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
