#!/usr/bin/env bash
# Refuse commits on the repo's default branch.
# Bypass with: SKIP=protect-default-branch git commit ...

set -euo pipefail

default=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@')
head=$(git symbolic-ref --short HEAD 2>/dev/null || true)

if [ "$head" = "$default" ]; then
  echo "ERROR: refusing to commit on $default"
  echo "  - create a feature branch:        git switch -c <name>"
  echo "  - intentional direct commit:      SKIP=protect-default-branch git commit ..."
  exit 1
fi
