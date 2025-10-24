# RAG 評估系統 v2.0 - 三層評估架構

## 🎯 系統特性

本系統採用**三層評估架構**，提供全方位的 RAG 系統品質評估：

### 📊 三層評估架構

```
┌─────────────────────────────────────────┐
│       RAG 評估系統 v2.0                  │
│     三層評估架構 + 靈活權重配置           │
└─────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
   自動評估           人工/AI評估
        │                 │
   ┌────┴────┐      ┌─────┴─────┐
   │         │      │           │
 第一層    第二層   第三層        │
關鍵詞    語義    GPT Judge    │
 匹配    相似度   (多維度)      │
   │         │      │           │
   └────┬────┴──────┴───────────┘
        │
   ┌────┴─────────────────────┐
   │    智能權重配置系統        │
   │  - 自動權重建議            │
   │  - 手動權重調整            │
   │  - 多場景適配              │
   └──────────────────────────┘
```

#### 第一層：關鍵詞匹配（必選）
- **評估內容**: 回答中包含的關鍵詞比例
- **特色**: 支援同義詞識別、專有名詞保護
- **速度**: 極快（< 1秒/題）
- **適用場景**: 快速篩選、基礎評估

#### 第二層：語義相似度（推薦）
- **評估內容**: 語義層面的匹配度
- **技術**: Sentence Transformers
- **速度**: 中等（1-2秒/題）
- **適用場景**: 全面評估、品質保證

#### 第三層：GPT as a Judge（深度分析）
- **評估內容**: 多維度深度分析
  - 相關性 (Relevance)
  - 完整性 (Completeness)
  - 準確性 (Accuracy)
  - 忠實度 (Faithfulness)
- **速度**: 較慢（5-10秒/題）
- **成本**: 約 $0.002/題（GPT-3.5-turbo）
- **適用場景**: 深度分析、質化研究

---

## 🚀 快速開始

### 1. 環境安裝

#### 最小安裝（僅第一層）
```bash
pip install pandas numpy jieba streamlit plotly openpyxl xlsxwriter
```

#### 推薦安裝（第一層 + 第二層）
```bash
pip install -r requirements_v2.txt
```

**注意**: 如果不需要 GPT 評審，可以跳過安裝 `openai` 套件。

### 2. 資料準備

#### 輸入資料格式

Excel 或 CSV 檔案需包含以下欄位：

| 欄位名稱 | 說明 | 必填 |
|---------|------|-----|
| 序號 | 問題編號 | 是 |
| 測試問題 | 測試的問題內容 | 是 |
| 應回答之詞彙 | 期望回答包含的關鍵詞 | 是 |
| 向量知識庫（原始版） | 原始版本的回答 | 是 |
| 智慧文檔知識庫（彙整版） | 優化版本的回答 | 是 |

**範例資料**:

```csv
序號,測試問題,應回答之詞彙,向量知識庫（原始版）,智慧文檔知識庫（彙整版）
1,請問工作許可證的申請流程為何？,工作許可證、申請流程、包商名稱,需要向工安部門申請...,工作許可證的申請流程包括...
```

#### 輸出資料格式

評估完成後會生成包含以下欄位的 Excel 檔案：

**基本資訊**:
- 序號、測試問題、應回答之詞彙

**原始版本評分**:
- KEYWORD_COVERAGE_ORIGINAL (關鍵詞覆蓋率)
- SEMANTIC_SIMILARITY_ORIGINAL (語義相似度)
- GPT_OVERALL_ORIGINAL (GPT 評分)
- FINAL_SCORE_ORIGINAL (綜合評分)

**優化版本評分**:
- KEYWORD_COVERAGE_OPTIMIZED (關鍵詞覆蓋率)
- SEMANTIC_SIMILARITY_OPTIMIZED (語義相似度)
- GPT_OVERALL_OPTIMIZED (GPT 評分)
- FINAL_SCORE_OPTIMIZED (綜合評分)

**改善分析**:
- KEYWORD_IMPROVEMENT (關鍵詞改善幅度)
- SEMANTIC_IMPROVEMENT (語義改善幅度)
- GPT_IMPROVEMENT (GPT 評分改善)
- FINAL_IMPROVEMENT (綜合改善幅度)

### 3. 啟動系統

#### 方式 1: Streamlit 網頁介面（推薦）

```bash
streamlit run streamlit_comparison_dashboard_v2.py
```

然後在瀏覽器中打開 `http://localhost:8501`

#### 方式 2: Python 腳本

```python
from rag_evaluation_two_models_v2 import RAGEvaluatorV2

# 基本使用（關鍵詞 + 語義）
evaluator = RAGEvaluatorV2(
    'test_data/測試資料.xlsx',
    model_type='cross',
    enable_semantic=True,
    enable_gpt=False
)

# 執行評估
results = evaluator.evaluate_all()

# 保存結果
evaluator.save_results('評估結果_v2.xlsx')

# 查看統計
stats = evaluator.generate_summary_stats()
print(stats)
```

---

## 📖 使用指南

### 評估層級配置

#### 場景 1: 快速評估（僅關鍵詞）
```python
evaluator = RAGEvaluatorV2(
    'data.xlsx',
    enable_semantic=False,
    enable_gpt=False
)
```
- **速度**: 最快
- **準確度**: 基礎
- **成本**: 免費
- **適用**: 快速篩選、初步評估

#### 場景 2: 平衡評估（關鍵詞 + 語義）【推薦】
```python
evaluator = RAGEvaluatorV2(
    'data.xlsx',
    enable_semantic=True,
    enable_gpt=False,
    weights={"keyword": 0.5, "semantic": 0.5, "gpt": 0.0}
)
```
- **速度**: 中等
- **準確度**: 良好
- **成本**: 免費
- **適用**: 大多數評估場景

#### 場景 3: 深度分析（全三層）
```python
evaluator = RAGEvaluatorV2(
    'data.xlsx',
    enable_semantic=True,
    enable_gpt=True,
    openai_api_key='your-api-key',
    weights={"keyword": 0.3, "semantic": 0.3, "gpt": 0.4}
)
```
- **速度**: 較慢
- **準確度**: 最佳
- **成本**: 約 $0.002/題
- **適用**: 深度分析、研究項目

### 權重配置建議

系統會根據啟用的評估層級**自動建議權重**，您也可以手動調整：

| 評估模式 | 關鍵詞 | 語義 | GPT | 說明 |
|---------|--------|------|-----|------|
| 單層（僅關鍵詞） | 1.0 | 0.0 | 0.0 | 快速評估 |
| 雙層（關鍵詞+語義） | 0.5 | 0.5 | 0.0 | 推薦配置 |
| 雙層（關鍵詞+GPT） | 0.4 | 0.0 | 0.6 | 深度但無語義 |
| 三層（全部） | 0.3 | 0.3 | 0.4 | 最全面 |

**自訂權重範例**:
```python
# 強調關鍵詞匹配
weights = {"keyword": 0.6, "semantic": 0.3, "gpt": 0.1}

# 強調 GPT 評審
weights = {"keyword": 0.2, "semantic": 0.2, "gpt": 0.6}
```

---

## 💡 最佳實踐

### 1. 資料品質
- ✅ 確保「應回答之詞彙」清晰明確
- ✅ 包含 20+ 個測試問題以獲得統計意義
- ✅ 問題應涵蓋不同難度和主題

### 2. 評估策略
- 🚀 **初期**: 使用關鍵詞匹配快速篩選
- 📊 **日常**: 使用關鍵詞+語義進行全面評估
- 🔬 **深度**: 對關鍵問題使用 GPT 評審深入分析

### 3. 成本控制
- 💰 GPT 評審成本: 100 題約 $0.20 USD
- 💡 建議: 先用語義相似度評估，對低分問題再用 GPT 深度分析
- 🎯 可以設定「僅對綜合評分 < 60 的問題啟用 GPT」

### 4. 結果解讀

#### 關鍵詞覆蓋率
- **≥ 80%**: 優秀（包含大部分關鍵資訊）
- **60-79%**: 良好（包含主要關鍵資訊）
- **< 60%**: 需改善（遺漏重要資訊）

#### 語義相似度
- **≥ 70%**: 語義高度相關
- **50-69%**: 語義部分相關
- **< 50%**: 語義不相關或偏離主題

#### GPT 評分
- **≥ 80**: 高品質回答
- **60-79**: 中等品質
- **< 60**: 需要改善

#### 綜合評分
根據權重配置自動計算，反映整體回答品質。

---

## 🔧 進階功能

### 1. 批次處理

```python
import os
from rag_evaluation_two_models_v2 import RAGEvaluatorV2

# 批次處理資料夾中的所有檔案
data_folder = 'test_data'
for filename in os.listdir(data_folder):
    if filename.endswith('.xlsx'):
        evaluator = RAGEvaluatorV2(
            os.path.join(data_folder, filename),
            enable_semantic=True
        )
        results = evaluator.evaluate_all()
        evaluator.save_results(f'results_{filename}')
```

### 2. 自訂同義詞字典

在 `rag_evaluation_two_models_v2.py` 中修改 `_is_similar_term` 方法：

```python
def _is_similar_term(self, keyword: str, answer: str) -> bool:
    """檢查是否有相似詞彙"""
    synonyms = {
        "包商": ["承包商", "廠商", "承攬商"],
        "負責人": ["主管", "管理人", "聯絡人"],
        # 新增您的同義詞
        "系統": ["平台", "應用程式", "軟體"],
    }
    # ... 其餘程式碼
```

### 3. 整合到 CI/CD 流程

```yaml
# .github/workflows/rag-evaluation.yml
name: RAG Quality Check

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements_v2.txt
      - name: Run evaluation
        run: python run_evaluation.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: results/
```

---

## 🐛 常見問題

### Q1: 語義相似度功能無法使用？

**A**: 確保安裝 `sentence-transformers`:
```bash
pip install sentence-transformers torch
```

首次使用會自動下載模型（約 400MB），請確保網路暢通。

### Q2: GPT 評審返回錯誤？

**A**: 檢查以下項目：
1. ✅ OpenAI API 金鑰是否正確
2. ✅ 帳戶是否有足夠額度
3. ✅ 網路連線是否正常
4. ✅ 是否超過 API 速率限制

### Q3: 評估速度太慢？

**A**: 優化建議：
- 🚀 停用不需要的評估層級
- 🚀 使用批次處理（系統已自動批次化）
- 🚀 考慮僅對部分問題啟用 GPT 評審

### Q4: 綜合評分與各層級分數不一致？

**A**: 這是正常現象。綜合評分 = 各層級分數 × 對應權重，受權重配置影響。

### Q5: 如何解讀「改善幅度」為負值？

**A**: 負值表示優化版本在該指標上反而退步了。可能原因：
- 回答更簡潔（關鍵詞覆蓋率下降）
- 過度保守（語義相似度下降）
- 需要檢視具體回答內容判斷是否真的退步

---

## 📊 輸出範例

### 統計摘要範例

```json
{
  "原始版本": {
    "平均關鍵詞覆蓋率": 64.6,
    "平均語義相似度": 58.3,
    "平均GPT評分": 0.0,
    "平均綜合評分": 61.5
  },
  "彙整優化版本": {
    "平均關鍵詞覆蓋率": 83.0,
    "平均語義相似度": 72.1,
    "平均GPT評分": 0.0,
    "平均綜合評分": 77.6
  },
  "改善效果": {
    "平均關鍵詞覆蓋率提升": 18.4,
    "平均語義相似度提升": 13.8,
    "平均綜合評分提升": 16.1,
    "顯著改善比例": 65.0
  }
}
```

---

## 🆚 版本差異

| 特性 | v1.0 | v2.0 |
|------|------|------|
| 評估層級 | 單層（關鍵詞 + 忠誠度） | 三層（關鍵詞 + 語義 + GPT） |
| 語義分析 | ❌ | ✅ |
| GPT 評審 | ❌ | ✅ |
| 權重配置 | 固定 50/50 | 靈活調整 |
| 多維度評估 | ❌ | ✅ |
| 成本預估 | - | ✅ |
| 向後相容 | - | ✅ 可讀取 v1.0 資料 |

---

## 📞 技術支援

- **問題回報**: 請在 GitHub Issues 中提交
- **功能建議**: 歡迎提交 Pull Request
- **使用諮詢**: 請參考本文件或範例程式碼

---

## 📄 授權

MIT License

---

## 🎉 致謝

感謝以下開源專案：
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API](https://openai.com/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)

---

**版本**: 2.0
**最後更新**: 2024年
**作者**: AI 知識庫優化團隊
