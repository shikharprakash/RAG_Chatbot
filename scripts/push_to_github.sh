#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/shikharprakash/RAG_Chatbot.git"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: Git is not installed or is not on PATH." >&2
  exit 1
fi
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: $ROOT_DIR is not inside a Git repository." >&2
  exit 1
fi

# Keep dotenv files out of Git; .env.example is the documented safe template.
if ! grep -qxF '.env' .gitignore || ! grep -qxF '.env.*' .gitignore || ! grep -qxF '!.env.example' .gitignore; then
  echo "Error: .gitignore must ignore .env and .env.* while allowing .env.example." >&2
  exit 1
fi

git add -A
git rm --cached --ignore-unmatch --quiet -- .env

if git diff --cached --name-only | grep -E '(^|/)\.env($|\.)' | grep -qv '^\.env\.example$'; then
  echo "Error: a dotenv file other than .env.example is staged; refusing to push." >&2
  exit 1
fi
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "Error: .env is tracked. It has been removed from the index; review and commit that change first." >&2
  exit 1
fi
if git log --all --format= -- .env | grep -q .; then
  echo "Error: .env exists in Git history. Refusing to push potentially exposed secrets." >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REPO_URL"
else
  current_remote="$(git remote get-url origin)"
  if [[ "$current_remote" != "$REPO_URL" ]]; then
    echo "Error: origin is configured for a different repository; refusing to overwrite it." >&2
    exit 1
  fi
fi

if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
  echo "Error: configure your Git author first (git config --global user.name and user.email)." >&2
  exit 1
fi

if ! git diff --cached --quiet; then
  git commit -m "Initial project commit"
fi

if [[ -z "$(git rev-parse --verify HEAD 2>/dev/null || true)" ]]; then
  echo "Error: there is no commit to push." >&2
  exit 1
fi
branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  echo "Error: detached HEAD; check out a branch before pushing." >&2
  exit 1
fi

credential_file="$(mktemp "${TMPDIR:-/tmp}/github-credentials.XXXXXX")"
chmod 600 "$credential_file"
cleanup() {
  rm -f "$credential_file"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -r /dev/tty ]]; then
  echo "Error: an interactive terminal is required to enter the token securely." >&2
  exit 1
fi
printf 'GitHub token (input hidden; do not paste it into chat): '
IFS= read -r -s token < /dev/tty
printf '\n'
if [[ -z "$token" ]]; then
  echo "Error: token cannot be empty." >&2
  exit 1
fi
printf 'https://x-access-token:%s@github.com\n' "$token" > "$credential_file"
unset token

printf 'Pushing branch %s to %s...\n' "$branch" "$REPO_URL"
git -c credential.helper= \
  -c "credential.helper=store --file=$credential_file" \
  push --set-upstream origin "$branch"
echo "Push completed. .env was excluded."
