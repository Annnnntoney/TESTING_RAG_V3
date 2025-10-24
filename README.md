# RAG 評估儀表板 v2（Manual GPT）

此專案是一個 Streamlit 儀表板，整合關鍵詞覆蓋率、語義相似度與人工 GPT 評審三層指標，協助比較原始版與優化版回答。核心程式為 `streamlit_dashboard_v2_with_manual_gpt.py`，其他模組提供運算與資料管理功能。

## 核心檔案

| 檔案 | 角色 |
| --- | --- |
| `streamlit_dashboard_v2_with_manual_gpt.py` | 主程式。包含上傳/評估流程、GPT JSON 貼上區、歷史紀錄、各頁籤 UI（評估總覽、GPT 分析、綜合篩選器…）。 |
| `rag_evaluation_two_models_v2.py` | 定義 `RAGEvaluatorV2`，負責載入測試檔案、計算關鍵詞覆蓋率與語義相似度等底層指標。語義模型需先安裝 `sentence-transformers`、`torch`。 |
| `evaluation_history_manager.py` | 管理 GPT 評分歷史：儲存/載入人工貼上的 JSON、產生 `llm_judge_table.csv`、匯出完整報告等。 |
| `combined_filter_tab.py` | 實作「🔎 綜合篩選器」頁籤，提供條件 slider、結果表格與 CSV 匯出，並讀取 `llm_judge_table.csv` 依當前 GPT 維度/權重重新計算分數。 |

## 安裝與執行

```bash
pip install -r requirements_v2.txt
streamlit run streamlit_dashboard_v2_with_manual_gpt.py
```

若需 GPT 評分輔助檔案，可參考 `gpt_manual_evaluation_helper.py`；所有指標說明與操作指南已整理於 `README_v2.md`、`GPT補充說明*.md` 等文件。

## 使用流程概述

1. 從側欄上傳測試結果（Excel/CSV）。
2. 儀表板計算關鍵詞與語義指標；至「GPT 人工評審」頁籤生成 Prompt，貼至 ChatGPT 後將 JSON 回貼並儲存。
3. 儲存後資料會寫入 `llm_judge_table.csv`，頁面上的 GPT 分數與理由同步更新。
4. 利用「🔎 綜合篩選器」設定條件快速找出需要關注的題目，或於「📥 下載結果」匯出完整報告。

> 若看到「語義相似度模型未啟動」提示，請先 `pip install sentence-transformers torch`，再重新啟動 Streamlit。
