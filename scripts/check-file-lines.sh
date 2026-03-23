#!/bin/sh
# Reject files exceeding line limits (src: 600, tests: 800)
status=0
for f in "$@"; do
  lines=$(wc -l < "$f")
  case "$f" in
    */tests/*) limit=800 ;;
    *)         limit=600 ;;
  esac
  if [ "$lines" -gt "$limit" ]; then
    echo "$f: $lines lines (limit: $limit)"
    status=1
  fi
done
exit $status
