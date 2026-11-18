#!/bin/bash
osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "cd /Users/vanch/VideoLingo-plus && /Users/vanch/pinokio/bin/miniconda/envs/videolingo/bin/python scripts/monitor_batch.py"
end tell
APPLESCRIPT
