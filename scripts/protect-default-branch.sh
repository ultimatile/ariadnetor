#!/usr/bin/env bash
# Refuse pushes to the repo's default branch.
# Bypass with: SKIP=protect-default-branch git push ...
#
# pre-commit's pre-push integration exposes the destination ref via
# PRE_COMMIT_REMOTE_BRANCH (e.g. "refs/heads/main"). The hook does not
# receive the raw pre-push stdin under the pre-commit framework, so
# parsing stdin (as a vanilla pre-push hook would) is incorrect here.
# Reference: https://pre-commit.com/#pre-commit-during-push

set -euo pipefail

default=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's@^origin/@@')
remote_branch="${PRE_COMMIT_REMOTE_BRANCH:-}"

if [ "$remote_branch" = "refs/heads/$default" ]; then
  echo "ERROR: refusing to push to $remote_branch"
  echo "  - intentional push: SKIP=protect-default-branch git push ..."
  exit 1
fi
