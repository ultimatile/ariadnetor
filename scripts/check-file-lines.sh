#!/bin/sh
# Reject files over line limits: 800 for tests (tests/ dirs and tests.rs submodules), 600 for other source
status=0
for f in "$@"; do
  lines=$(wc -l < "$f")
  case "$f" in
    */tests/*|*/tests.rs) limit=800 ;;
    *)         limit=600 ;;
  esac
  if [ "$lines" -gt "$limit" ]; then
    echo "$f: $lines lines (limit: $limit)"
    status=1
  fi
done
exit $status
