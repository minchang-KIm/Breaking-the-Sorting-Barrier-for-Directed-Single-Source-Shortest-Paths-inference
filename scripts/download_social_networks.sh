#!/bin/bash
################################################################################
# 소셜 네트워크 데이터셋 다운로드 스크립트
# Social Network Dataset Download Script
#
# Stanford SNAP - Large Network Dataset Collection
# Source: https://snap.stanford.edu/data/
#
# 사용법 / Usage:
#   ./scripts/download_social_networks.sh [all|small|medium|large]
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
DATASET_DIR="datasets/social_networks"
mkdir -p "$DATASET_DIR"

echo "================================================================================"
echo "  소셜 네트워크 데이터셋 다운로드"
echo "  Social Network Dataset Download"
echo "================================================================================"
echo ""

MODE="${1:-small}"

case "$MODE" in
    all)
        log_info "모든 소셜 네트워크 다운로드 / Downloading all social networks"
        DATASETS=("wiki-Vote" "email-Enron" "web-Google" "roadNet-CA")
        ;;
    small)
        log_info "소규모 소셜 네트워크 다운로드 / Downloading small social networks"
        DATASETS=("wiki-Vote" "email-Enron")
        ;;
    medium)
        log_info "중규모 소셜 네트워크 다운로드 / Downloading medium social networks"
        DATASETS=("web-Google" "roadNet-CA")
        ;;
    large)
        log_info "대규모 소셜 네트워크 다운로드 / Downloading large social networks"
        log_warn "Twitter-2010은 매우 큼 (35GB), 수동 다운로드 권장"
        DATASETS=()
        ;;
    *)
        log_error "알 수 없는 모드: $MODE"
        exit 1
        ;;
esac

echo ""

# SNAP 데이터셋 URL
declare -A DATASET_URLS
DATASET_URLS[wiki-Vote]="https://snap.stanford.edu/data/wiki-Vote.txt.gz"
DATASET_URLS[email-Enron]="https://snap.stanford.edu/data/email-Enron.txt.gz"
DATASET_URLS[web-Google]="https://snap.stanford.edu/data/web-Google.txt.gz"
DATASET_URLS[roadNet-CA]="https://snap.stanford.edu/data/roadNet-CA.txt.gz"

# 데이터셋 정보
declare -A DATASET_INFO
DATASET_INFO[wiki-Vote]="7115:103689:Wikipedia Vote Network"
DATASET_INFO[email-Enron]="36692:367662:Enron Email Network"
DATASET_INFO[web-Google]="875713:5105039:Google Web Graph"
DATASET_INFO[roadNet-CA]="1965206:5533214:California Road Network"

# 함수: SNAP 형식을 우리 형식으로 변환
convert_snap_to_our_format() {
    local input_file=$1
    local output_file=$2
    local vertices=$3
    local edges=$4

    log_info "변환 중 / Converting: $input_file -> $output_file"

    # SNAP 형식: FromNodeId  ToNodeId
    # 우리 형식: n m
    #           u1 v1 w1
    #           ...

    # 헤더 작성
    echo "$vertices $edges" > "$output_file"

    # 데이터 변환 (주석 제거, 가중치 1.0 추가)
    grep -v '^#' "$input_file" | awk '{print $1, $2, 1.0}' >> "$output_file"

    log_success "변환 완료 / Conversion completed"
}

# 함수: 데이터셋 다운로드
download_dataset() {
    local name=$1
    local url=${DATASET_URLS[$name]}
    local info=${DATASET_INFO[$name]}

    IFS=':' read -r vertices edges description <<< "$info"

    log_info "다운로드 중 / Downloading: $description"
    log_info "  URL: $url"
    log_info "  정점 수 / Vertices: $(printf "%'d" $vertices)"
    log_info "  간선 수 / Edges: $(printf "%'d" $edges)"

    local gz_file="$DATASET_DIR/${name}.txt.gz"
    local txt_file="$DATASET_DIR/${name}_snap.txt"
    local output_file="$DATASET_DIR/${name}.txt"

    # 이미 변환된 파일 존재하면 스킵
    if [ -f "$output_file" ]; then
        log_warn "이미 존재함, 스킵 / Already exists, skipping: $output_file"
        return 0
    fi

    # wget이나 curl로 다운로드
    if command -v wget &> /dev/null; then
        log_info "wget으로 다운로드 중..."
        wget -q --show-progress -O "$gz_file" "$url" || {
            log_error "다운로드 실패 / Download failed"
            return 1
        }
    elif command -v curl &> /dev/null; then
        log_info "curl로 다운로드 중..."
        curl -L -o "$gz_file" "$url" --progress-bar || {
            log_error "다운로드 실패 / Download failed"
            return 1
        }
    else
        log_error "wget 또는 curl이 필요합니다 / wget or curl required"
        log_info "수동 다운로드: $url"
        return 1
    fi

    # 압축 해제
    log_info "압축 해제 중 / Decompressing..."
    gunzip -c "$gz_file" > "$txt_file"
    rm "$gz_file"

    # 형식 변환
    convert_snap_to_our_format "$txt_file" "$output_file" "$vertices" "$edges"

    # 임시 파일 삭제
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
log_success "소셜 네트워크 다운로드 완료! / Social network download completed!"
echo "================================================================================"
echo ""
log_info "데이터셋 위치 / Dataset location: $DATASET_DIR"
log_info "다운로드된 파일 목록 / Downloaded files:"
ls -lh "$DATASET_DIR"/*.txt 2>/dev/null || log_warn "파일 없음 / No files found"

echo ""
log_info "추가 데이터셋 / Additional datasets:"
echo "  Stanford SNAP: https://snap.stanford.edu/data/"
echo "  Twitter-2010 (41M vertices, 1.4B edges, 35GB):"
echo "    https://snap.stanford.edu/data/twitter-2010.html"

exit 0
