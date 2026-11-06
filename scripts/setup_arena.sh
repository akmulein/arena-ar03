#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
AR3_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
CASE_SRC="$AR3_ROOT/arena-case"
ARENA_URL="https://gitlab.deeppavlov.ai/deeppavlov/arena.git"
# Pinned from the local Arena clone used for the verified AR-03 runs.
ARENA_COMMIT="669d9ed31840bfc46be10c8c63ccd8c3cd1f76ab"
DEFAULT_ARENA_DIR="$HOME/.cache/ar03-arena"
ARENA_DIR="${ARENA_DIR:-$DEFAULT_ARENA_DIR}"
REBUILD_IMAGES=0
REPLACE_CASE=0

usage() {
    cat <<EOF
Usage: scripts/setup_arena.sh [options]

Clone or reuse DeepPavlov Arena, pin commit $ARENA_COMMIT,
sync its dependencies, link arena-case/ as cases/ar-03/,
and build the AR-03 Docker images if they are missing.

Options:
  --arena-dir PATH   Arena clone path (default: $DEFAULT_ARENA_DIR)
  --replace-case     Move an existing cases/ar-03 aside and link it again
  --rebuild-images   Rebuild ar03-runner:local and ar03-metric:local
  -h, --help         Show this help
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --arena-dir)
            [ "$#" -ge 2 ] || die "missing value for --arena-dir"
            ARENA_DIR="$2"
            shift 2
            ;;
        --arena-dir=*)
            ARENA_DIR="${1#*=}"
            shift
            ;;
        --replace-case)
            REPLACE_CASE=1
            shift
            ;;
        --rebuild-images)
            REBUILD_IMAGES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (try --help)"
            ;;
    esac
done

[ -n "$ARENA_DIR" ] || die "--arena-dir must not be empty"
[ "$ARENA_DIR" != "/" ] || die "refusing to use / as the Arena directory"

case "$ARENA_DIR" in
    "~") ARENA_DIR="$HOME" ;;
    "~/"*) ARENA_DIR="$HOME/${ARENA_DIR#~/}" ;;
esac
case "$ARENA_DIR" in
    /*) ;;
    *) ARENA_DIR="$(pwd -P)/$ARENA_DIR" ;;
esac
case "/$ARENA_DIR/" in
    *"/../"*) die "Arena directory must not contain '..'" ;;
esac

if [ "$ARENA_DIR" = "$AR3_ROOT" ]; then
    die "Arena directory must be outside AR-3: $ARENA_DIR"
fi
case "$ARENA_DIR/" in
    "$AR3_ROOT/"*) die "Arena directory must be outside AR-3: $ARENA_DIR" ;;
esac

ARENA_PARENT="$(dirname "$ARENA_DIR")"
if [ "$ARENA_PARENT" = "$AR3_ROOT" ]; then
    die "Arena directory must be outside AR-3: $ARENA_DIR"
fi
case "$ARENA_PARENT/" in
    "$AR3_ROOT/"*) die "Arena directory must be outside AR-3: $ARENA_DIR" ;;
esac

mkdir -p "$ARENA_PARENT"
ARENA_DIR="$(cd "$ARENA_PARENT" && pwd -P)/$(basename "$ARENA_DIR")"

if [ "$ARENA_DIR" = "$AR3_ROOT" ]; then
    die "Arena directory must be outside AR-3: $ARENA_DIR"
fi
case "$ARENA_DIR/" in
    "$AR3_ROOT/"*) die "Arena directory must be outside AR-3: $ARENA_DIR" ;;
esac
if [ -d "$ARENA_DIR" ]; then
    ARENA_REAL="$(cd "$ARENA_DIR" && pwd -P)"
    case "$ARENA_REAL/" in
        "$AR3_ROOT/"*) die "Arena directory resolves inside AR-3: $ARENA_REAL" ;;
    esac
fi

[ -d "$CASE_SRC" ] || die "arena-case/ not found at $CASE_SRC"
[ -f "$CASE_SRC/container/Dockerfile" ] || die "container/Dockerfile not found in $CASE_SRC"
[ -f "$CASE_SRC/container_check.py" ] || die "container_check.py not found in $CASE_SRC"
[ -f "$CASE_SRC/metric/score.py" ] || die "metric/score.py not found in $CASE_SRC"
[ -f "$CASE_SRC/ways/baseline/solve.py" ] || die "ways/baseline/solve.py not found in $CASE_SRC"

command -v git >/dev/null 2>&1 || die "git is required"
command -v uv >/dev/null 2>&1 || die "uv is required"
command -v docker >/dev/null 2>&1 || die "docker is required"

if command -v python3 >/dev/null 2>&1; then
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
        echo "warning: system python3 is older than 3.12; uv may provision a compatible interpreter"
    fi
else
    echo "warning: python3 not found; uv will manage the interpreter"
fi

docker info >/dev/null 2>&1 || die "Docker daemon is not available"

if [ -e "$ARENA_DIR" ] && [ ! -d "$ARENA_DIR" ]; then
    die "$ARENA_DIR exists but is not a directory"
fi

clone_arena() {
    echo "Cloning Arena into $ARENA_DIR"
    git clone "$ARENA_URL" "$ARENA_DIR" || die "failed to clone Arena from $ARENA_URL"
}

if [ -d "$ARENA_DIR" ] && git -C "$ARENA_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    echo "Using existing Arena clone: $ARENA_DIR"
    if [ -n "$(git -C "$ARENA_DIR" status --porcelain --untracked-files=no)" ]; then
        die "Arena clone has uncommitted tracked changes; commit or stash them first"
    fi
elif [ -d "$ARENA_DIR" ] && [ -z "$(ls -A "$ARENA_DIR")" ]; then
    clone_arena
elif [ -e "$ARENA_DIR" ]; then
    die "$ARENA_DIR exists but is not a Git clone; choose another --arena-dir"
else
    clone_arena
fi

if ! git -C "$ARENA_DIR" cat-file -e "${ARENA_COMMIT}^{commit}" 2>/dev/null; then
    echo "Pinned commit not found locally; fetching origin"
    if ! git -C "$ARENA_DIR" fetch --tags origin; then
        die "failed to fetch Arena; commit $ARENA_COMMIT is not available"
    fi
fi
git -C "$ARENA_DIR" cat-file -e "${ARENA_COMMIT}^{commit}" 2>/dev/null \
    || die "Arena commit $ARENA_COMMIT is not available in $ARENA_DIR"

git -C "$ARENA_DIR" checkout --detach "$ARENA_COMMIT" \
    || die "failed to checkout Arena commit $ARENA_COMMIT"
[ "$(git -C "$ARENA_DIR" rev-parse HEAD)" = "$ARENA_COMMIT" ] \
    || die "failed to pin Arena to $ARENA_COMMIT"

CASE_DST="$ARENA_DIR/cases/ar-03"
backup_case() {
    local backup="${CASE_DST}.backup.$$-$(date +%s)"
    echo "Moving existing $CASE_DST to $backup"
    mv "$CASE_DST" "$backup"
}
mkdir -p "$ARENA_DIR/cases"
if [ -L "$CASE_DST" ]; then
    if [ ! -e "$CASE_DST" ]; then
        [ "$REPLACE_CASE" -eq 1 ] || die "$CASE_DST is a broken symlink; rerun with --replace-case"
        backup_case
        ln -s "$CASE_SRC" "$CASE_DST"
    else
        CASE_TARGET="$(cd "$CASE_DST" && pwd -P)"
        if [ "$CASE_TARGET" != "$CASE_SRC" ]; then
            [ "$REPLACE_CASE" -eq 1 ] \
                || die "$CASE_DST points to $CASE_TARGET; rerun with --replace-case"
            backup_case
            ln -s "$CASE_SRC" "$CASE_DST"
        fi
    fi
elif [ -e "$CASE_DST" ]; then
    [ "$REPLACE_CASE" -eq 1 ] \
        || die "$CASE_DST already exists and is not a symlink; move it aside or rerun with --replace-case"
    backup_case
    ln -s "$CASE_SRC" "$CASE_DST"
else
    ln -s "$CASE_SRC" "$CASE_DST"
fi
[ -d "$CASE_DST" ] || die "failed to link $CASE_SRC as $CASE_DST"

[ -f "$ARENA_DIR/uv.lock" ] || die "uv.lock not found in $ARENA_DIR"
echo "Syncing Arena dependencies with uv sync --frozen"
(cd "$ARENA_DIR" && uv sync --frozen)

build_image() {
    local image="$1"
    shift
    if [ "$REBUILD_IMAGES" -eq 0 ] && docker image inspect "$image" >/dev/null 2>&1; then
        echo "Docker image already present, skipping: $image"
        return 0
    fi
    echo "Building Docker image: $image"
    docker build -t "$image" "$@" || die "Docker build failed: $image"
}

build_image ar03-runner:local \
    --build-context "arena_runner=$ARENA_DIR/runner" \
    --target runner \
    -f "$CASE_SRC/container/Dockerfile" \
    "$CASE_SRC"
build_image ar03-metric:local \
    --target metric \
    -f "$CASE_SRC/container/Dockerfile" \
    "$CASE_SRC"

if [ ! -f "$AR3_ROOT/data/train/manifest.jsonl" ]; then
    echo "warning: $AR3_ROOT/data/train/manifest.jsonl not found; the container check needs the local train dataset"
fi

cat <<EOF

Arena is ready: $ARENA_DIR
Case link: $CASE_DST -> $CASE_SRC

Smoke test:
  cd "$ARENA_DIR"
  uv run python -m evaluator.smoke

AR-03 container check (requires local train data):
  cd "$ARENA_DIR"
  uv run python -m cases.ar-03.container_check --train "$AR3_ROOT/data/train"

API submit -> score and evolution are not checked by this script.
EOF
