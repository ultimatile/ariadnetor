#!/usr/bin/env bash
# Refuse pushes to the repo's default branch.
# Bypass with: SKIP=protect-default-branch git push ...

set -euo pipefail

default=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@')

while read -r _local_ref _local_sha remote_ref _remote_sha; do
  if [ "$remote_ref" = "refs/heads/$default" ]; then
    echo "ERROR: refusing to push to $remote_ref"
    echo "  - intentional push: SKIP=protect-default-branch git push ..."
    exit 1
  fi
done
