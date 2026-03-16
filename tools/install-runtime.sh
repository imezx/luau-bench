#!/usr/bin/env bash

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$INSTALL_DIR"

LUAU_VERSION=""
ZUNE_VERSION=""
STYLUA_VERSION=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_err()  { echo -e "${RED}✗${NC} $1"; }
log_info() { echo -e "${CYAN}->${NC} $1"; }

detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)       echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)  echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *)             echo "unknown" ;;
    esac
}

OS=$(detect_os)
ARCH=$(detect_arch)

gh_latest_tag() {
    local repo="$1"
    local redirect
    redirect=$(curl -fsSI --max-time 10 "https://github.com/${repo}/releases/latest" \
               -o /dev/null -w '%{redirect_url}' 2>/dev/null)
    if [[ -z "$redirect" ]]; then
        log_err "Could not resolve latest release for ${repo}."
        log_err "Check your internet connection or pass an explicit version flag."
        return 1
    fi
    echo "${redirect##*/}"
}

download() {
    local url="$1"
    local dest="$2"
    if command -v curl &>/dev/null; then
        curl -fsSL "$url" -o "$dest"
    elif command -v wget &>/dev/null; then
        wget -q "$url" -O "$dest"
    else
        log_err "Neither curl nor wget is available."
        return 1
    fi
}


install_luau() {
    echo ""
    echo "Installing Luau..."

    if command -v luau &>/dev/null && [[ -z "$LUAU_VERSION" ]]; then
        log_warn "luau already installed: $(which luau)  (pass --luau-version to upgrade)"
        return
    fi

    local tag
    if [[ -n "$LUAU_VERSION" ]]; then
        tag="$LUAU_VERSION"
        log_info "Using requested version: ${tag}"
    else
        log_info "Resolving latest Luau version..."
        tag=$(gh_latest_tag "luau-lang/luau") || return 1
        tag="${tag#v}"
        log_info "Latest: ${tag}"
    fi

    local asset_os=""
    case "$OS" in
        linux) asset_os="ubuntu" ;;
        macos) asset_os="macos" ;;
        *)
            log_err "Unsupported OS for pre-built Luau: ${OS}"
            log_err "Build from source: https://github.com/luau-lang/luau"
            return 1
            ;;
    esac

    local asset_name="luau-${asset_os}.zip"
    local url="https://github.com/luau-lang/luau/releases/download/${tag}/${asset_name}"
    log_info "Downloading ${asset_name} from ${url}"

    if download "$url" /tmp/luau.zip 2>/dev/null; then
        unzip -o /tmp/luau.zip -d "$INSTALL_DIR"
        chmod +x "$INSTALL_DIR/luau" "$INSTALL_DIR/luau-analyze" 2>/dev/null || true
        rm -f /tmp/luau.zip

        if "$INSTALL_DIR/luau" --help &>/dev/null; then
            log_ok "Luau ${tag} installed -> ${INSTALL_DIR}/luau"
            return
        fi
        log_warn "Binary extracted but failed to execute — trying source build."
    else
        log_warn "Pre-built binary download failed — trying source build."
    fi

    if command -v cmake &>/dev/null && command -v g++ &>/dev/null; then
        echo "  Building from source (this may take a few minutes)..."
        local tmpdir
        tmpdir=$(mktemp -d)
        local clone_args=(--depth 1)
        if [[ -n "$LUAU_VERSION" ]]; then
            clone_args+=(--branch "$LUAU_VERSION")
        fi
        git clone "${clone_args[@]}" https://github.com/luau-lang/luau.git "$tmpdir/luau"
        cd "$tmpdir/luau"
        cmake . -DCMAKE_BUILD_TYPE=Release -B build
        cmake --build build --target Luau.Repl.CLI --config Release -j"$(nproc 2>/dev/null || echo 4)"
        cp build/luau "$INSTALL_DIR/luau"
        chmod +x "$INSTALL_DIR/luau"
        rm -rf "$tmpdir"
        log_ok "Luau ${tag} installed from source -> ${INSTALL_DIR}/luau"
    else
        log_err "Cannot install Luau. Options:"
        echo "  1. Install cmake + g++:  sudo apt install cmake g++ git"
        echo "  2. Download manually:    https://github.com/luau-lang/luau/releases/tag/${tag}"
        return 1
    fi
}


install_zune() {
    echo ""
    echo "Installing Zune runtime..."

    if command -v zune &>/dev/null && [[ -z "$ZUNE_VERSION" ]]; then
        log_warn "zune already installed: $(which zune)  (pass --zune-version to upgrade)"
        return
    fi

    local tag
    if [[ -n "$ZUNE_VERSION" ]]; then
        tag="v${ZUNE_VERSION#v}"
        log_info "Using requested version: ${tag}"
    else
        log_info "Resolving latest Zune version..."
        tag=$(gh_latest_tag "Scythe-Technology/zune") || return 1
        log_info "Latest: ${tag}"
    fi
    local version="${tag#v}"

    local zune_os zune_arch
    case "$OS" in
        linux) zune_os="linux" ;;
        macos) zune_os="macos" ;;
        *)     log_err "Unsupported OS: ${OS}"; return 1 ;;
    esac
    case "$ARCH" in
        x86_64)  zune_arch="x86_64" ;;
        aarch64) zune_arch="aarch64" ;;
        *)       log_err "Unsupported arch: ${ARCH}"; return 1 ;;
    esac

    local asset_name="zune-${version}-${zune_os}-${zune_arch}.zip"
    local url="https://github.com/Scythe-Technology/zune/releases/download/${tag}/${asset_name}"
    log_info "Downloading ${asset_name} from ${url}"

    download "$url" /tmp/zune.zip || {
        log_err "Download failed."
        echo "  Install manually: https://github.com/Scythe-Technology/zune/releases/tag/${tag}"
        return 1
    }

    unzip -o /tmp/zune.zip -d "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR/zune"
    rm -f /tmp/zune.zip

    if "$INSTALL_DIR/zune" --version &>/dev/null; then
        log_ok "Zune ${version} installed -> ${INSTALL_DIR}/zune"
    else
        log_warn "Binary extracted but may not be compatible with this system."
    fi
}

install_stylua() {
    echo ""
    echo "Installing StyLua..."

    if command -v stylua &>/dev/null && [[ -z "$STYLUA_VERSION" ]]; then
        log_warn "stylua already installed: $(which stylua)  (pass --stylua-version to upgrade)"
        return
    fi

    if command -v cargo &>/dev/null && [[ -z "$STYLUA_VERSION" ]]; then
        cargo install stylua --features luau
        log_ok "StyLua installed via cargo"
        return
    fi

    local tag
    if [[ -n "$STYLUA_VERSION" ]]; then
        tag="v${STYLUA_VERSION#v}"
        log_info "Using requested version: ${tag}"
    else
        log_info "Resolving latest StyLua version..."
        tag=$(gh_latest_tag "JohnnyMorganz/StyLua") || return 1
        log_info "Latest: ${tag}"
    fi

    local stylua_platform
    case "$OS-$ARCH" in
        linux-x86_64)  stylua_platform="linux-x86_64" ;;
        linux-aarch64) stylua_platform="linux-aarch64" ;;
        macos-x86_64)  stylua_platform="macos-x86_64" ;;
        macos-aarch64) stylua_platform="macos-aarch64" ;;
        *)
            log_err "Unsupported platform: ${OS}-${ARCH}"
            return 1
            ;;
    esac

    local asset_name="stylua-${stylua_platform}.zip"
    local url="https://github.com/JohnnyMorganz/StyLua/releases/download/${tag}/${asset_name}"
    log_info "Downloading ${asset_name} from ${url}"

    download "$url" /tmp/stylua.zip || {
        log_err "Download failed."
        echo "  Install manually: https://github.com/JohnnyMorganz/StyLua/releases/tag/${tag}"
        return 1
    }

    unzip -o /tmp/stylua.zip -d "$INSTALL_DIR"
    chmod +x "$INSTALL_DIR/stylua"
    rm -f /tmp/stylua.zip

    if "$INSTALL_DIR/stylua" --version &>/dev/null; then
        log_ok "StyLua ${tag} installed -> ${INSTALL_DIR}/stylua"
    else
        log_warn "Binary extracted but may not be compatible with this system."
    fi
}

show_help() {
    cat << 'HELP'
Usage: tools/install-runtime.sh [tools] [version options]

Tools (pick one or more):
  --luau          Install the Luau CLI and luau-analyze
  --zune          Install the Zune runtime
  --stylua        Install StyLua formatter (with Luau support)
  --all           Install all three

Version options (default: latest release):
  --luau-version VERSION      Luau release tag without 'v', e.g. 0.712
  --zune-version VERSION      Zune release tag without 'v', e.g. 0.1.0
  --stylua-version VERSION    StyLua release tag without 'v', e.g. 2.0.2

Environment:
  INSTALL_DIR     Installation directory (default: ~/.local/bin)

Examples:
  ./tools/install-runtime.sh --all
  ./tools/install-runtime.sh --luau --luau-version 0.712
  ./tools/install-runtime.sh --zune --zune-version 0.2.0
  INSTALL_DIR=/usr/local/bin ./tools/install-runtime.sh --all

After installation, verify with:
  luau-bench info
HELP
}

if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

DO_LUAU=false
DO_ZUNE=false
DO_STYLUA=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --luau)    DO_LUAU=true ;;
        --zune)    DO_ZUNE=true ;;
        --stylua)  DO_STYLUA=true ;;
        --all)
            DO_LUAU=true
            DO_ZUNE=true
            DO_STYLUA=true
            ;;
        --luau-version)
            shift
            LUAU_VERSION="${1:?'--luau-version requires a value'}"
            ;;
        --zune-version)
            shift
            ZUNE_VERSION="${1:?'--zune-version requires a value'}"
            ;;
        --stylua-version)
            shift
            STYLUA_VERSION="${1:?'--stylua-version requires a value'}"
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_err "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

if ! $DO_LUAU && ! $DO_ZUNE && ! $DO_STYLUA; then
    log_err "No tools selected. Use --luau, --zune, --stylua, or --all."
    show_help
    exit 1
fi

$DO_LUAU   && install_luau
$DO_ZUNE   && install_zune
$DO_STYLUA && install_stylua

echo ""
echo "Done. Make sure ${INSTALL_DIR} is in your PATH:"
echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
echo ""
echo "Verify with:  luau-bench info"
echo ""
