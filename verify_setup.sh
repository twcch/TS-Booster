#!/bin/bash
# 快速驗證腳本 - 確認 paper_booster 環境正常

echo "=================================="
echo "TS-Booster 環境驗證"
echo "=================================="

# 1. 檢查目錄結構
echo ""
echo "1. 檢查目錄結構..."
if [ -d "src" ] && [ -d "experiments" ] && [ -d "dataset" ]; then
    echo "   ✅ 目錄結構正確"
else
    echo "   ❌ 目錄結構不完整"
    exit 1
fi

# 2. 檢查核心檔案
echo ""
echo "2. 檢查核心檔案..."
files=(
    "src/models/flexible_booster.py"
    "src/models/ts_booster.py"
    "src/models/layers.py"
    "experiments/comprehensive_evaluation.py"
    "experiments/visualize_comprehensive.py"
    "dataset/USD_TWD.csv"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (缺失)"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    exit 1
fi

# 3. 檢查 Python 環境
echo ""
echo "3. 檢查 Python 環境..."
if command -v uv &> /dev/null; then
    echo "   ✅ uv 已安裝"
else
    echo "   ⚠️  uv 未安裝 (可使用 pip 替代)"
fi

# 4. 測試導入
echo ""
echo "4. 測試模組導入..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from src.models.flexible_booster import FlexibleTSBooster
    from src.models.layers import LinearBackbone
    print('   ✅ 模組導入成功')
except Exception as e:
    print(f'   ❌ 模組導入失敗: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# 5. 檢查實驗結果
echo ""
echo "5. 檢查實驗結果..."
if [ -d "experiments/comprehensive_results/figures" ]; then
    num_figures=$(ls experiments/comprehensive_results/figures/*.png 2>/dev/null | wc -l)
    if [ $num_figures -gt 0 ]; then
        echo "   ✅ 找到 $num_figures 張圖表"
    else
        echo "   ⚠️  尚未生成圖表"
    fi
else
    echo "   ⚠️  尚未運行實驗"
fi

echo ""
echo "=================================="
echo "✅ 驗證完成!"
echo "=================================="
echo ""
echo "下一步:"
echo "1. 安裝依賴: uv sync"
echo "2. 運行實驗: uv run python experiments/comprehensive_evaluation.py --experiment all --quick"
echo ""
