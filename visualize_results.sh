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
    echo "  --oracle              - 顯示 Oracle 分析報告"
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
SEEN_PAIRS=()
ABLATION_FILES=()
ABLATION_FILE=""
CROSS_BACKBONE_FILE=""
VISUALIZE_ALL=false
SHOW_ORACLE=false

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
        --oracle)
            SHOW_ORACLE=true
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

# 如果使用最新檔案,自動尋找（每個 dataset 取最新的一個）
if [ "$USE_LATEST" = true ]; then
    if [ -z "$ABLATION_FILE" ]; then
        # 找出所有 dataset，每個取最新的檔案
        ABLATION_FILES=()
        for f in $(ls -t $RESULTS_DIR/ablation_results_*.json 2>/dev/null); do
            # 從檔名提取 currency pair (e.g., USD_TWD)
            pair=$(echo "$(basename "$f")" | sed -n 's/ablation_results_\(.*\)_[0-9]\{8\}_.*/\1/p')
            # 只保留每個 pair 的第一個（最新的）
            if [[ ! " ${SEEN_PAIRS[*]} " =~ " ${pair} " ]]; then
                SEEN_PAIRS+=("$pair")
                ABLATION_FILES+=("$f")
            fi
        done
        # 向下相容：如果只有一個檔案，也設定 ABLATION_FILE
        if [ ${#ABLATION_FILES[@]} -eq 1 ]; then
            ABLATION_FILE="${ABLATION_FILES[0]}"
        elif [ ${#ABLATION_FILES[@]} -gt 1 ]; then
            ABLATION_FILE="__MULTI__"
        fi
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

if [ "$ABLATION_FILE" = "__MULTI__" ]; then
    HAS_ABLATION=true
    echo -e "${GREEN}✓${NC} 找到 ${#ABLATION_FILES[@]} 個 Ablation 結果:"
    for f in "${ABLATION_FILES[@]}"; do
        echo "    - $(basename "$f")"
    done
elif [ -n "$ABLATION_FILE" ] && [ -f "$ABLATION_FILE" ]; then
    HAS_ABLATION=true
    ABLATION_FILES=("$ABLATION_FILE")
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
    for ablation_f in "${ABLATION_FILES[@]}"; do
        pair=$(echo "$(basename "$ablation_f")" | sed -n 's/ablation_results_\([A-Z]*_TWD\)_.*/\1/p')
        echo -e "${YELLOW}📊 生成 Ablation Study 圖表 (${pair})...${NC}"
        uv run python experiments/visualize_comprehensive.py \
            --ablation "$ablation_f" \
            --results-dir "$RESULTS_DIR"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ ${pair} Ablation 圖表生成完成${NC}"
        else
            echo -e "${RED}❌ ${pair} Ablation 圖表生成失敗${NC}"
        fi
    done

    # Cross-dataset comparison (if multiple datasets)
    if [ ${#ABLATION_FILES[@]} -ge 2 ]; then
        echo -e "${YELLOW}📊 生成跨數據集比較圖表...${NC}"
        uv run python experiments/visualize_comprehensive.py \
            --cross-dataset \
            --results-dir "$RESULTS_DIR"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ 跨數據集比較圖表生成完成${NC}"
        else
            echo -e "${RED}❌ 跨數據集比較圖表生成失敗${NC}"
        fi
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

# 顯示 Oracle 分析 (如果請求)
if [ "$SHOW_ORACLE" = true ]; then
    echo -e "${YELLOW}📊 Oracle 分析報告${NC}"
    echo ""

    ORACLE_REPORT=$(find_latest_file "oracle_analysis_*.md")

    if [ -n "$ORACLE_REPORT" ] && [ -f "$ORACLE_REPORT" ]; then
        echo -e "${GREEN}✓${NC} 找到 Oracle 分析: $(basename $ORACLE_REPORT)"
        echo ""
        cat "$ORACLE_REPORT"
        echo ""
    else
        echo -e "${YELLOW}⚠ 找不到 Oracle 分析報告${NC}"
        echo "請先運行: ./run_experiment.sh oracle"
        echo ""
    fi
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
    for ablation_f in "${ABLATION_FILES[@]}"; do
        ABLATION_REPORT="${ablation_f%.json}.md"
        echo "  cat $ABLATION_REPORT"
    done
fi
if [ "$HAS_CROSS_BACKBONE" = true ]; then
    CROSS_BACKBONE_REPORT="${CROSS_BACKBONE_FILE%.json}.md"
    echo "  cat $CROSS_BACKBONE_REPORT"
fi
echo ""
echo "查看 Oracle 分析:"
echo "  ./visualize_results.sh --oracle"
echo ""
