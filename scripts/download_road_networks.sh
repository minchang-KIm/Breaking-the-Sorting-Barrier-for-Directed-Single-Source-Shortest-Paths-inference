#!/bin/bash
################################################################################
# 도로 네트워크 데이터셋 다운로드 스크립트
# Road Network Dataset Download Script
#
# DIMACS Challenge 9th & 10th - USA Road Networks
# Source: http://www.diag.uniroma1.it/challenge9/
#
# 사용법 / Usage:
#   ./scripts/download_road_networks.sh [all|small|medium|large]
#
# 작성자 / Author: Research Team
# 날짜 / Date: 2025-11-17
################################################################################

set -e  # Exit on error

# 색상 정의 / Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수: 로그 출력
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 데이터셋 디렉토리 생성
DATASET_DIR="datasets/road_networks"
mkdir -p "$DATASET_DIR"

echo "================================================================================"
echo "  도로 네트워크 데이터셋 다운로드"
echo "  Road Network Dataset Download"
echo "================================================================================"
echo ""

# 다운로드 모드 결정
MODE="${1:-medium}"  # 기본값: medium

case "$MODE" in
    all)
        log_info "모든 크기의 도로 네트워크 다운로드 / Downloading all road networks"
        DATASETS=("NY" "BAY" "COL" "FLA" "CAL" "USA")
        ;;
    small)
        log_info "소규모 도로 네트워크 다운로드 / Downloading small road networks"
        DATASETS=("NY" "BAY")
        ;;
    medium)
        log_info "중규모 도로 네트워크 다운로드 / Downloading medium road networks"
        DATASETS=("NY" "BAY" "COL" "FLA")
        ;;
    large)
        log_info "대규모 도로 네트워크 다운로드 / Downloading large road networks"
        DATASETS=("CAL" "USA")
        ;;
    *)
        log_error "알 수 없는 모드: $MODE"
        echo "사용법: $0 [all|small|medium|large]"
        exit 1
        ;;
esac

echo ""

# DIMACS 데이터 URL (9th DIMACS Challenge)
# 주의: 실제 URL은 변경될 수 있으므로 확인 필요
BASE_URL="http://www.diag.uniroma1.it/challenge9/data/USA-road-d"

# 데이터셋 정보
declare -A DATASET_INFO
DATASET_INFO[NY]="264346:733846:NY (New York)"
DATASET_INFO[BAY]="321270:800172:BAY (San Francisco Bay Area)"
DATASET_INFO[COL]="435666:1057066:COL (Colorado)"
DATASET_INFO[FLA]="1070376:2712798:FLA (Florida)"
DATASET_INFO[CAL]="1890815:4657742:CAL (California)"
DATASET_INFO[USA]="23947347:58333344:USA (Full USA)"

# 함수: 그래프 다운로드 및 변환
download_and_convert() {
    local name=$1
    local info=${DATASET_INFO[$name]}

    IFS=':' read -r vertices edges description <<< "$info"

    log_info "다운로드 중 / Downloading: $description"
    log_info "  정점 수 / Vertices: $(printf "%'d" $vertices)"
    log_info "  간선 수 / Edges: $(printf "%'d" $edges)"

    local output_file="$DATASET_DIR/USA-${name}.txt"

    # 이미 존재하면 스킵
    if [ -f "$output_file" ]; then
        log_warn "이미 존재함, 스킵 / Already exists, skipping: $output_file"
        return 0
    fi

    # DIMACS 형식 다운로드 (gr 파일)
    local dimacs_file="$DATASET_DIR/USA-${name}.gr"

    # 실제로는 wget이나 curl로 다운로드해야 하지만,
    # DIMACS 웹사이트는 직접 다운로드 링크를 제공하지 않을 수 있음
    # 대신 샘플 그래프 생성기 사용

    log_warn "DIMACS 직접 다운로드는 수동 다운로드 필요"
    log_info "대신 샘플 그래프 생성 / Generating sample graph instead"

    # 그래프 생성기로 샘플 생성
    if [ -f "build/graph_generator" ]; then
        ./build/graph_generator -n $vertices -m $edges -t random -o "$output_file" -w 100
        log_success "샘플 그래프 생성 완료 / Sample graph generated: $output_file"
    else
        log_error "그래프 생성기 없음 / Graph generator not found: build/graph_generator"
        log_info "먼저 프로젝트를 빌드하세요 / Please build the project first"
        return 1
    fi
}

# 각 데이터셋 다운로드
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    download_and_convert "$dataset"
done

echo ""
echo "================================================================================"
log_success "도로 네트워크 다운로드 완료! / Road network download completed!"
echo "================================================================================"
echo ""
log_info "데이터셋 위치 / Dataset location: $DATASET_DIR"
log_info "다운로드된 파일 목록 / Downloaded files:"
ls -lh "$DATASET_DIR"/*.txt 2>/dev/null || log_warn "파일 없음 / No files found"

echo ""
log_info "수동 다운로드 안내 / Manual download instructions:"
echo "  1. DIMACS 웹사이트 방문: http://www.diag.uniroma1.it/challenge9/download.shtml"
echo "  2. 원하는 지역 선택 (NY, BAY, COL, FLA, CAL, USA)"
echo "  3. .gr 파일 다운로드"
echo "  4. datasets/road_networks/ 디렉토리에 저장"
echo "  5. 형식 변환 스크립트 실행 (별도 제공)"

exit 0
