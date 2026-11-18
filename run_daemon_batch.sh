#!/bin/bash
cd /Users/vanch/VideoLingo-plus
export PYTHONUNBUFFERED=1
exec /Users/vanch/pinokio/bin/miniconda/envs/videolingo/bin/python -u -m core.cli batch --profile fast >> batch/batch_processor.log 2>&1
