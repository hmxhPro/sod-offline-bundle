#!/bin/bash
# Backend startup script with offline mode for HuggingFace

# Force HuggingFace to use local cache only
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Silence FFmpeg / libav decoder chatter (SEI truncated, NAL warnings, etc.)
# These are non-fatal H.264 stream artifacts that flood stderr but do not
# affect detection. -8 = AV_LOG_QUIET.
export OPENCV_FFMPEG_LOGLEVEL=-8
export AV_LOG_FORCE_NOCOLOR=1

# Start the backend server
cd "$(dirname "$0")"

# Dev convenience: DEV_RELOAD=1 enables uvicorn auto-reload on app/*.py changes,
# so backend code edits take effect without a manual restart. OFF by default
# (production). NOTE: --reload restarts the server whenever a watched file
# changes, which would interrupt an in-flight training run — leave it off while
# training. Changes to .env or YOLOWorld/ are NOT watched (restart for those).
RELOAD_ARGS=()
if [ "${DEV_RELOAD:-0}" = "1" ]; then
  RELOAD_ARGS=(--reload --reload-dir app)
  echo "[start.sh] DEV_RELOAD=1 → uvicorn auto-reload enabled (watching app/)"
fi

# --timeout-graceful-shutdown: give the lifespan shutdown hook time to drain
# in-flight detection tasks and reap training subprocess groups (R-2).
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-graceful-shutdown 30 "${RELOAD_ARGS[@]}"
