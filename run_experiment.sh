#!/bin/bash
# TS-Booster 實驗執行腳本
# 使用方式: ./run_experiment.sh [experiment_type] [options]

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
    echo "TS-Booster 實驗執行腳本"
    echo -e "==================================${NC}"
    echo ""
    echo "使用方式:"
    echo "  ./run_experiment.sh [experiment_type] [options]"
    echo ""
    echo "實驗類型:"
    echo "  ablation          - 運行 Ablation Study (4 種配置)"
    echo "  cross-backbone    - 運行 Cross-Backbone Validation (4 種骨幹)"
    echo "  all               - 運行所有實驗 (消融 + 骨幹)"
    echo ""
    echo "選項:"
    echo "  --quick          - 快速測試 (20 epochs)"
    echo "  --epochs N       - 指定訓練輪數 (預設: 100)"
    echo "  --dataset NAME   - 指定數據集 (預設: USD_TWD)"
    echo "  --seed N         - 隨機種子 (預設: 42)"
    echo "  --keep-old       - 保留舊結果 (預設: 自動清理)"
    echo ""
    echo "範例:"
    echo "  ./run_experiment.sh ablation --quick"
    echo "  ./run_experiment.sh all --epochs 100"
    echo "  ./run_experiment.sh cross-backbone --dataset JPY_TWD --seed 123"
    echo "  ./run_experiment.sh all --epochs 100 --dataset JPY_TWD"
    echo ""
}

# 檢查參數
if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
    exit 0
fi

EXPERIMENT_TYPE=$1
shift  # 移除第一個參數

# 預設值
EPOCHS=100
QUICK_MODE=""
DATASET="USD_TWD"
SEED=42
KEEP_OLD=""

# 解析選項
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE="--quick"
            EPOCHS=20
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            QUICK_MODE=""
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --keep-old)
            KEEP_OLD="--keep-old"
            shift
            ;;
        *)
            echo -e "${RED}未知選項: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# 顯示執行資訊
echo -e "${BLUE}=================================="
echo "開始執行實驗"
echo -e "==================================${NC}"
echo -e "${YELLOW}實驗類型:${NC} $EXPERIMENT_TYPE"
echo -e "${YELLOW}訓練輪數:${NC} $EPOCHS epochs"

echo -e "${YELLOW}數據集:${NC} $DATASET"

echo -e "${YELLOW}隨機種子:${NC} $SEED"
if [ -n "$QUICK_MODE" ]; then
    echo -e "${YELLOW}模式:${NC} 快速測試"
fi
if [ -n "$KEEP_OLD" ]; then
    echo -e "${YELLOW}舊結果:${NC} 保留"
else
    echo -e "${YELLOW}舊結果:${NC} 自動清理"
fi
echo ""

# 執行實驗
case $EXPERIMENT_TYPE in
    ablation)
        echo -e "${GREEN}運行 Ablation Study...${NC}"
        if [ -n "$QUICK_MODE" ]; then
            uv run python experiments/comprehensive_evaluation.py --experiment ablation --quick --dataset "dataset/${DATASET}.csv" --seed "$SEED" $KEEP_OLD
        else
            uv run python experiments/comprehensive_evaluation.py --experiment ablation --epochs "$EPOCHS" --dataset "dataset/${DATASET}.csv" --seed "$SEED" $KEEP_OLD
        fi
        ;;
    cross-backbone)
        echo -e "${GREEN}運行 Cross-Backbone Validation...${NC}"
        if [ -n "$QUICK_MODE" ]; then
            uv run python experiments/comprehensive_evaluation.py --experiment cross-backbone --quick --dataset "dataset/${DATASET}.csv" --seed "$SEED" $KEEP_OLD
        else
            uv run python experiments/comprehensive_evaluation.py --experiment cross-backbone --epochs "$EPOCHS" --dataset "dataset/${DATASET}.csv" --seed "$SEED" $KEEP_OLD
        fi
        ;;
    all)
        echo -e "${GREEN}運行所有實驗...${NC}"
        if [ -n "$QUICK_MODE" ]; then
            uv run python experiments/comprehensive_evaluation.py --experiment all --quick --dataset "dataset/${DATASET}.csv" --seed "$SEED" $KEEP_OLD
        else
            uv run python experiments/comprehensive_evaluation.py --experiment all --epochs "$EPOCHS" --dataset "dataset/${DATASET}.csv" --seed "$SEED" $KEEP_OLD
        fi
        ;;
    *)
        echo -e "${RED}錯誤: 未知的實驗類型 '$EXPERIMENT_TYPE'${NC}"
        show_usage
        exit 1
        ;;
esac

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================="
    echo "✅ 實驗完成!"
    echo -e "==================================${NC}"
    echo ""
    echo "下一步:"
    echo "  1. 查看結果: ls experiments/comprehensive_results/"
    echo "  2. 生成圖表: ./visualize_results.sh"
    echo ""
else
    echo ""
    echo -e "${RED}=================================="
    echo "❌ 實驗失敗!"
    echo -e "==================================${NC}"
    exit 1
fi
