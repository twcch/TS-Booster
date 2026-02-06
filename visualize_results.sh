#!/bin/bash
# TS-Booster 結果視覺化腳本
# 使用方式: ./visualize_results.sh [options]

set -e  # 遇到錯誤立即停止

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 顯示使用說明
show_usage() {
    echo -e "${BLUE}=================================="
    echo "TS-Booster 結果視覺化腳本"
    echo -e "==================================${NC}"
    echo ""
    echo "使用方式:"
    echo "  ./visualize_results.sh [options]"
    echo ""
    echo "選項:"
    echo "  --ablation FILE       - 指定 Ablation 結果檔案"
    echo "  --cross-backbone FILE - 指定 Cross-Backbone 結果檔案"
    echo "  --latest              - 自動使用最新的結果檔案 (預設)"
    echo "  --all                 - 視覺化所有結果"
    echo ""
    echo "範例:"
    echo "  ./visualize_results.sh --latest"
    echo "  ./visualize_results.sh --all"
    echo "  ./visualize_results.sh --ablation experiments/comprehensive_results/ablation_results_*.json"
    echo ""
}

# 檢查參數
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
    exit 0
fi

# 結果目錄
RESULTS_DIR="experiments/comprehensive_results"

# 尋找最新的結果檔案
find_latest_file() {
    local pattern=$1
    local latest=$(ls -t $RESULTS_DIR/$pattern 2>/dev/null | head -1)
    echo "$latest"
}

# 預設使用最新檔案
USE_LATEST=true
ABLATION_FILE=""
CROSS_BACKBONE_FILE=""
VISUALIZE_ALL=false

# 解析選項
while [[ $# -gt 0 ]]; do
    case $1 in
        --ablation)
            ABLATION_FILE="$2"
            USE_LATEST=false
            shift 2
            ;;
        --cross-backbone)
            CROSS_BACKBONE_FILE="$2"
            USE_LATEST=false
            shift 2
            ;;
        --latest)
            USE_LATEST=true
            shift
            ;;
        --all)
            VISUALIZE_ALL=true
            USE_LATEST=true
            shift
            ;;
        *)
            echo -e "${RED}未知選項: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# 如果使用最新檔案,自動尋找
if [ "$USE_LATEST" = true ]; then
    if [ -z "$ABLATION_FILE" ]; then
        ABLATION_FILE=$(find_latest_file "ablation_results_*.json")
    fi
    if [ -z "$CROSS_BACKBONE_FILE" ]; then
        CROSS_BACKBONE_FILE=$(find_latest_file "cross_backbone_results_*.json")
    fi
fi

# 顯示執行資訊
echo -e "${BLUE}=================================="
echo "開始生成視覺化"
echo -e "==================================${NC}"
echo ""

# 檢查是否有結果檔案
HAS_ABLATION=false
HAS_CROSS_BACKBONE=false

if [ -n "$ABLATION_FILE" ] && [ -f "$ABLATION_FILE" ]; then
    HAS_ABLATION=true
    echo -e "${GREEN}✓${NC} 找到 Ablation 結果: $(basename $ABLATION_FILE)"
fi

if [ -n "$CROSS_BACKBONE_FILE" ] && [ -f "$CROSS_BACKBONE_FILE" ]; then
    HAS_CROSS_BACKBONE=true
    echo -e "${GREEN}✓${NC} 找到 Cross-Backbone 結果: $(basename $CROSS_BACKBONE_FILE)"
fi

if [ "$HAS_ABLATION" = false ] && [ "$HAS_CROSS_BACKBONE" = false ]; then
    echo -e "${RED}❌ 錯誤: 找不到任何結果檔案${NC}"
    echo ""
    echo "請先運行實驗:"
    echo "  ./run_experiment.sh ablation"
    echo "  ./run_experiment.sh cross-backbone"
    exit 1
fi

echo ""

# 生成 Ablation 視覺化
if [ "$HAS_ABLATION" = true ]; then
    echo -e "${YELLOW}📊 生成 Ablation Study 圖表...${NC}"
    uv run python experiments/visualize_comprehensive.py \
        --ablation "$ABLATION_FILE" \
        --results-dir "$RESULTS_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Ablation 圖表生成完成${NC}"
        echo "   - ablation_progression.png"
        echo "   - ablation_all_metrics.png"
        echo "   - ablation_improvement.png"
    else
        echo -e "${RED}❌ Ablation 圖表生成失敗${NC}"
    fi
    echo ""
fi

# 生成 Cross-Backbone 視覺化
if [ "$HAS_CROSS_BACKBONE" = true ]; then
    echo -e "${YELLOW}📊 生成 Cross-Backbone 圖表...${NC}"
    uv run python experiments/visualize_comprehensive.py \
        --cross-backbone "$CROSS_BACKBONE_FILE" \
        --results-dir "$RESULTS_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Cross-Backbone 圖表生成完成${NC}"
        echo "   - backbone_comparison.png"
        echo "   - backbone_improvement.png"
    else
        echo -e "${RED}❌ Cross-Backbone 圖表生成失敗${NC}"
    fi
    echo ""
fi

# 顯示完成訊息
echo -e "${GREEN}=================================="
echo "✅ 視覺化完成!"
echo -e "==================================${NC}"
echo ""
echo "圖表位置: $RESULTS_DIR/figures/"
echo ""
echo "查看圖表:"
echo "  ls $RESULTS_DIR/figures/"
echo ""
echo "查看報告:"
if [ "$HAS_ABLATION" = true ]; then
    ABLATION_REPORT="${ABLATION_FILE%.json}.md"
    echo "  cat $ABLATION_REPORT"
fi
if [ "$HAS_CROSS_BACKBONE" = true ]; then
    CROSS_BACKBONE_REPORT="${CROSS_BACKBONE_FILE%.json}.md"
    echo "  cat $CROSS_BACKBONE_REPORT"
fi
echo ""
