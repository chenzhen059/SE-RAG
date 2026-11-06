#!/usr/bin/env bash

set -u

usage() {
  cat <<'EOF'
Usage: auto_git_sync.sh [--once] [--watch] [--dry-run]

Synchronizes a Git repository once by default, then exits.
Use --watch to keep polling for changes.

Environment variables:
  AUTO_GIT_REPO             Repository path. Defaults to this script's repo.
  AUTO_GIT_REMOTE           Remote name. Defaults to origin.
  AUTO_GIT_BRANCH           Branch name. Defaults to current branch.
  AUTO_GIT_INTERVAL         Watch-mode poll interval in seconds. Defaults to 30.
  AUTO_GIT_DEBOUNCE         Wait time after detecting changes. Defaults to 60.
  AUTO_GIT_LOG              Log file. Defaults to .auto-git-sync.log.
  AUTO_GIT_ADD_UNTRACKED    Set to 1 to include untracked files. Defaults to 0.
  AUTO_GIT_COMMIT_PREFIX    Commit message prefix. Defaults to auto-sync.

By default this only stages tracked file changes with `git add -u`.
New files are committed only if you already staged them, unless
AUTO_GIT_ADD_UNTRACKED=1 is set.
EOF
}

log() {
  local line
  line="$(printf '%s %s' "$(date '+%F %T')" "$*")"

  printf '%s\n' "$line"
  printf '%s\n' "$line" >>"$LOG_FILE" 2>/dev/null || true
}

repo_status() {
  local untracked_mode="no"

  if [[ "${AUTO_GIT_ADD_UNTRACKED:-0}" == "1" ]]; then
    untracked_mode="all"
  fi

  git -C "$REPO_DIR" status --porcelain --untracked-files="$untracked_mode"
}

has_changes() {
  [[ -n "$(repo_status)" ]]
}

has_unpushed_commits() {
  local branch="$1"
  local upstream_ref="${AUTO_GIT_REMOTE}/${branch}"
  local count

  if ! git -C "$REPO_DIR" rev-parse --verify --quiet "$upstream_ref" >/dev/null; then
    return 0
  fi

  count="$(git -C "$REPO_DIR" rev-list --count "$upstream_ref..HEAD" 2>/dev/null || printf '1')"
  [[ "$count" != "0" ]]
}

current_branch() {
  if [[ -n "${AUTO_GIT_BRANCH:-}" ]]; then
    printf '%s\n' "$AUTO_GIT_BRANCH"
  else
    git -C "$REPO_DIR" branch --show-current
  fi
}

ensure_git_identity() {
  if ! git -C "$REPO_DIR" config --get user.name >/dev/null; then
    log "Git user.name is not configured. Run: git config user.name 'Your Name'"
    return 1
  fi

  if ! git -C "$REPO_DIR" config --get user.email >/dev/null; then
    log "Git user.email is not configured. Run: git config user.email 'you@example.com'"
    return 1
  fi
}

stage_changes() {
  if [[ "${AUTO_GIT_ADD_UNTRACKED:-0}" == "1" ]]; then
    git -C "$REPO_DIR" add -A
  else
    git -C "$REPO_DIR" add -u
  fi
}

push_branch() {
  local branch="$1"

  if ! has_unpushed_commits "$branch"; then
    log "No local commits to push."
    return 0
  fi

  if ! git -C "$REPO_DIR" push "$AUTO_GIT_REMOTE" "$branch" >>"$LOG_FILE" 2>&1; then
    log "Push failed. Check GitHub credentials or remote permissions."
    return 1
  fi

  log "Pushed commit $(git -C "$REPO_DIR" rev-parse --short HEAD) to ${AUTO_GIT_REMOTE}/${branch}."
}

sync_once() {
  local branch commit_message
  branch="$(current_branch)"

  if [[ -z "$branch" ]]; then
    log "Detached HEAD detected; skipping auto-sync."
    return 0
  fi

  if ! has_changes; then
    if [[ "$DRY_RUN" == "1" ]]; then
      if has_unpushed_commits "$branch"; then
        log "Dry run: would push local commits to ${AUTO_GIT_REMOTE}/${branch}."
      else
        log "Dry run: no tracked changes or local commits to sync."
      fi
      return 0
    fi

    log "No tracked changes detected; checking for local commits to push."
    push_branch "$branch"
    return 0
  fi

  log "Changes detected; waiting ${AUTO_GIT_DEBOUNCE}s before syncing."
  sleep "$AUTO_GIT_DEBOUNCE"

  if ! has_changes; then
    log "No changes remain after debounce; skipping."
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry run: these changes would be synced:"
    while IFS= read -r status_line; do
      log "  $status_line"
    done < <(repo_status)
    log "Dry run: would stage, commit, pull --rebase, and push to ${AUTO_GIT_REMOTE}/${branch}."
    return 0
  fi

  if ! ensure_git_identity; then
    return 1
  fi

  if ! stage_changes >>"$LOG_FILE" 2>&1; then
    log "Failed to stage changes."
    return 1
  fi

  if git -C "$REPO_DIR" diff --cached --quiet; then
    log "No staged changes after staging; skipping commit."
    push_branch "$branch"
    return $?
  fi

  commit_message="${AUTO_GIT_COMMIT_PREFIX}: $(date '+%Y-%m-%d %H:%M:%S')"
  if ! git -C "$REPO_DIR" commit -m "$commit_message" >>"$LOG_FILE" 2>&1; then
    log "Commit failed."
    return 1
  fi

  if ! git -C "$REPO_DIR" pull --rebase --autostash "$AUTO_GIT_REMOTE" "$branch" >>"$LOG_FILE" 2>&1; then
    log "Pull/rebase failed. Resolve conflicts manually, then rerun the script."
    return 1
  fi

  push_branch "$branch"
}

acquire_lock() {
  LOCK_DIR="$REPO_DIR/.auto-git-sync.lock"

  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" >"$LOCK_DIR/pid"
  elif [[ -f "$LOCK_DIR/pid" ]] && kill -0 "$(cat "$LOCK_DIR/pid")" 2>/dev/null; then
    log "Another auto-sync process is already running; exiting."
    exit 0
  else
    log "Removing stale auto-sync lock."
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || {
      log "Could not remove stale auto-sync lock; exiting."
      exit 1
    }
    mkdir "$LOCK_DIR" || exit 1
    printf '%s\n' "$$" >"$LOCK_DIR/pid"
  fi

  trap 'rm -f "$LOCK_DIR/pid"; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM
}

ONCE=0
DRY_RUN=0
WATCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      ONCE=1
      ;;
    --watch)
      WATCH=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_DIR="${AUTO_GIT_REPO:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)}"

if [[ -z "$REPO_DIR" || ! -d "$REPO_DIR/.git" ]]; then
  printf 'Could not find a Git repository. Set AUTO_GIT_REPO=/path/to/repo.\n' >&2
  exit 1
fi

AUTO_GIT_REMOTE="${AUTO_GIT_REMOTE:-origin}"
AUTO_GIT_INTERVAL="${AUTO_GIT_INTERVAL:-30}"
AUTO_GIT_DEBOUNCE="${AUTO_GIT_DEBOUNCE:-60}"
AUTO_GIT_COMMIT_PREFIX="${AUTO_GIT_COMMIT_PREFIX:-auto-sync}"
LOG_FILE="${AUTO_GIT_LOG:-$REPO_DIR/.auto-git-sync.log}"

export GIT_TERMINAL_PROMPT=0

if [[ "$DRY_RUN" != "1" ]]; then
  acquire_lock
fi

if [[ "$ONCE" == "1" || "$WATCH" != "1" || "$DRY_RUN" == "1" ]]; then
  sync_once
  exit $?
fi

log "Auto-sync watch started for $REPO_DIR. Polling every ${AUTO_GIT_INTERVAL}s."

while true; do
  sync_once
  sleep "$AUTO_GIT_INTERVAL"
done
