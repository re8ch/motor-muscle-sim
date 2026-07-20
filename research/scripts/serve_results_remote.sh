#!/usr/bin/env bash
set -euo pipefail

host="${1:-amdwsl-zt}"
port="${2:-8765}"
remote_root="/root/projects/motor-muscle-sim/research"

rsync -az research/viewer/ "${host}:${remote_root}/viewer/"
rsync -az \
  research/results/success_rate.png \
  research/results/teacher_smoke.mp4 \
  "${host}:${remote_root}/results/"

ssh "${host}" bash -s -- "${remote_root}" "${port}" <<'REMOTE'
set -euo pipefail
root="$1"
port="$2"
pidfile="${root}/results-viewer.pid"
logfile="${root}/results-viewer.log"

if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
  kill "$(cat "${pidfile}")"
fi

nohup python3 -m http.server "${port}" \
  --bind 0.0.0.0 \
  --directory "${root}" \
  >"${logfile}" 2>&1 &
echo $! >"${pidfile}"
sleep 1
curl --fail --silent "http://127.0.0.1:${port}/viewer/" >/dev/null
REMOTE

reachable_host="$(ssh -G "${host}" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }')"
echo "http://${reachable_host}:${port}/viewer/"
