#!/bin/bash
# Re-download the 6-clip test corpus (the pod that held them was shut down). yt-dlp from Dailymotion /
# archive.org (YouTube blocks datacenter IPs). First MAXS seconds are what run_bakeoff processes anyway.
set -eo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
D=${1:-/root/tracking/videos}; mkdir -p "$D"; cd "$D"
pip install -q -U yt-dlp 2>/dev/null || true
dl(){ yt-dlp -f "mp4/best" -o "$2.%(ext)s" "$1" || echo "FAILED: $2 ($1)"; }
dl "https://www.dailymotion.com/video/x95zq22" dm_alidawah_catholic_woman
dl "https://www.dailymotion.com/video/x9i50xa" dm_young_visitors_speakers_corner
dl "https://www.dailymotion.com/video/x9hqpp8" dm_dutchman_converts
dl "https://www.dailymotion.com/video/x2hzipv" dm_munadi_boat_basin
# archive.org daytime Speakers Corner clips (Dawah Wise / One Message) — set IA_1/IA_2 to the item mp4 URLs if needed
[ -n "$IA_1" ] && dl "$IA_1" ia_why_arent_you_muslim_dawahwise
[ -n "$IA_2" ] && dl "$IA_2" ia_shaykh_uthman_australian
# the original trial clip travels with the repo; copy it in if present
[ -f /root/offlineblur/videos/ali-dawah-street-interview-source.mp4 ] && cp /root/offlineblur/videos/ali-dawah-street-interview-source.mp4 .
ls -la "$D"
