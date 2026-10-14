#!/bin/sh
set -eu

BASE_URL=${BASE_URL:-http://localhost:80}

echo "GET /health"
curl --fail --silent --show-error "$BASE_URL/health"
printf '\n\n'

echo "POST /predict (valid payload)"
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"age":0.0380759064,"sex":0.0506801187,"bmi":0.0616962065,"bp":0.0218723855,"s1":-0.0442234984,"s2":-0.0348207628,"s3":-0.0434008457,"s4":-0.002592262,"s5":0.0199084209,"s6":-0.0176461252}' \
  "$BASE_URL/predict"
printf '\n\n'

echo "POST /predict (missing s6; expecting HTTP 422)"
response=$(curl --silent --show-error \
  --write-out '\n%{http_code}' \
  -H 'Content-Type: application/json' \
  -d '{"age":0.0380759064,"sex":0.0506801187,"bmi":0.0616962065,"bp":0.0218723855,"s1":-0.0442234984,"s2":-0.0348207628,"s3":-0.0434008457,"s4":-0.002592262,"s5":0.0199084209}' \
  "$BASE_URL/predict")
printf '%s\n' "$response"
status=$(printf '%s\n' "$response" | tail -n 1)
if [ "$status" != "422" ]; then
  echo "Expected HTTP 422 for invalid input." >&2
  exit 1
fi
