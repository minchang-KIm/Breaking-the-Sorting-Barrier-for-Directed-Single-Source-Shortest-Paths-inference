#!/bin/bash
################################################################################
# 웹 그래프 데이터셋 다운로드 스크립트
# Web Graph Dataset Download Script
#
# Stanford Web Graph Collection
# Source: https://snap.stanford.edu/data/
#
# 사용법 / Usage:
#   ./scripts/download_web_graphs.sh [all|small|medium]
#
# 작성자 / Author: Research Team
# 날짜 / Date: 2025-11-17
################################################################################

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 데이터셋 디렉토리
DATASET_DIR="datasets/web_graphs"
mkdir -p "$DATASET_DIR"

echo "================================================================================"
echo "  웹 그래프 데이터셋 다운로드"
echo "  Web Graph Dataset Download"
echo "================================================================================"
echo ""

MODE="${1:-small}"

case "$MODE" in
    all)
        DATASETS=("web-NotreDame" "web-Stanford" "web-BerkStan")
        ;;
    small)
        DATASETS=("web-NotreDame")
        ;;
    medium)
        DATASETS=("web-Stanford" "web-BerkStan")
        ;;
    *)
        log_error "알 수 없는 모드: $MODE"
        exit 1
        ;;
esac

# SNAP 웹 그래프 URL
declare -A DATASET_URLS
DATASET_URLS[web-NotreDame]="https://snap.stanford.edu/data/web-NotreDame.txt.gz"
DATASET_URLS[web-Stanford]="https://snap.stanford.edu/data/web-Stanford.txt.gz"
DATASET_URLS[web-BerkStan]="https://snap.stanford.edu/data/web-BerkStan.txt.gz"

# 데이터셋 정보
declare -A DATASET_INFO
DATASET_INFO[web-NotreDame]="325729:1497134:Notre Dame Web Graph"
DATASET_INFO[web-Stanford]="281903:2312497:Stanford Web Graph"
DATASET_INFO[web-BerkStan]="685230:7600595:Berkeley-Stanford Web Graph"

# 형식 변환 함수
convert_snap_to_our_format() {
    local input_file=$1
    local output_file=$2
    local vertices=$3
    local edges=$4

    log_info "변환 중 / Converting format..."

    echo "$vertices $edges" > "$output_file"
    grep -v '^#' "$input_file" | awk '{print $1, $2, 1.0}' >> "$output_file"

    log_success "변환 완료 / Conversion completed"
}

# 다운로드 함수
download_dataset() {
    local name=$1
    local url=${DATASET_URLS[$name]}
    local info=${DATASET_INFO[$name]}

    IFS=':' read -r vertices edges description <<< "$info"

    log_info "다운로드 중 / Downloading: $description"
    log_info "  정점 수 / Vertices: $(printf "%'d" $vertices)"
    log_info "  간선 수 / Edges: $(printf "%'d" $edges)"

    local gz_file="$DATASET_DIR/${name}.txt.gz"
    local txt_file="$DATASET_DIR/${name}_snap.txt"
    local output_file="$DATASET_DIR/${name}.txt"

    if [ -f "$output_file" ]; then
        log_warn "이미 존재함, 스킵 / Already exists: $output_file"
        return 0
    fi

    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$gz_file" "$url" || {
            log_error "다운로드 실패 / Download failed"
            return 1
        }
    elif command -v curl &> /dev/null; then
        curl -L -o "$gz_file" "$url" --progress-bar || {
            log_error "다운로드 실패 / Download failed"
            return 1
        }
    else
        log_error "wget 또는 curl 필요 / wget or curl required"
        return 1
    fi

    log_info "압축 해제 중 / Decompressing..."
    gunzip -c "$gz_file" > "$txt_file"
    rm "$gz_file"

    convert_snap_to_our_format "$txt_file" "$output_file" "$vertices" "$edges"
    rm "$txt_file"

    log_success "완료 / Completed: $output_file"
}

# 각 데이터셋 다운로드
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    download_dataset "$dataset" || log_warn "스킵됨 / Skipped: $dataset"
done

echo ""
echo "================================================================================"
log_success "웹 그래프 다운로드 완료! / Web graph download completed!"
echo "================================================================================"
echo ""
log_info "데이터셋 위치 / Dataset location: $DATASET_DIR"
ls -lh "$DATASET_DIR"/*.txt 2>/dev/null || log_warn "파일 없음 / No files found"

exit 0
