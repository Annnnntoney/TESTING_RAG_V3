# Streamlit Dashboard v2.0 ‑ Manual GPT Evaluation

這個儀表板是目前的主要入口 (`streamlit_dashboard_v2_with_manual_gpt.py`)。它把自動化指標與人工 GPT 評審整合在同一介面，協助你用一致的標準檢視 RAG 系統的回答。

## 核心特色

- **三層評估架構**：關鍵詞覆蓋率 + 語義相似度 + GPT as a Judge。
- **手動 GPT 評審流程**：儀表板產生標準化 Prompt，貼到 ChatGPT 後再把 JSON 回傳；系統會驗證欄位與數值，確保評分一致。
- **八個分析分頁**：
  1. 📊 評估總覽 – 加權總分與關鍵指標卡片。
  2. 🤖 GPT 人工評審 – Prompt 產生、評分貼回、指標說明。
  3. 📈 綜合比較圖表 – Radar/Bar 等視覺化比較。
  4. 🔤 語義分析 – 逐題語義差異與句子相似度。
  5. 💬 GPT 分析 – GPT 評分細節、改進建議。
  6. 🎯 關鍵詞分析 – 命中/缺漏關鍵詞與覆蓋率趨勢。
  7. 📥 下載結果 – 匯出 Excel / JSON 報告。
  8. 📝 GPT 補充說明 – 直接呈現 `GPT補充說明.md` 的操作指南。

## 評分方法摘要

| 指標 | 參考文獻 | 評估流程 |
| --- | --- | --- |
| 🎯 相關性 | Zheng et al., 2023, *LLMScore* | 拆句後標記 Strictly On-Topic / Off-Topic，列清單、算貼題比例。 |
| 📋 完整性 | Honovich et al., 2022, *TrueFewShot*；Liu et al., 2023, *MT-Bench* | 檢查【必須包含的關鍵資訊】，標記 Covered / Partially / Missing，計算命中率。 |
| ✅ 準確性 | Gao et al., 2023, *RLAIF*；Min et al., 2023, *FactScore* | 列出可驗證陳述，標記 Correct / Incorrect / Unverifiable，計算正確率。 |
| 🔒 忠誠度 | Maynez et al., 2020；Ji et al., 2023 | 標記 Supported / Partially Supported / Unsupported，計算支撐率。 |

每個指標在 Prompt 中都有 10% 等寬分級，並要求評審在 reasoning 中以人類語氣列出貼題／缺漏／正誤／支撐的句子與比例，最後用一句話總結扣分原因。

## 安裝與啟動

```bash
# 推薦在虛擬環境中執行
pip install -r requirements_v2.txt

# 先確認語義模型可載入（第一次會下載模型）
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
PY

# 啟動儀表板
streamlit run streamlit_dashboard_v2_with_manual_gpt.py
```

重新啟動前請用 `Ctrl+C` 停掉舊行程；若看到「⚠️ 語義相似度模型未啟動」，代表載入失敗或環境未安裝 `sentence-transformers`，請先完成上方檢查步驟再重跑。

## GPT 人工評審流程

1. 在 **🤖 GPT 人工評審** 分頁選題並複製 Prompt。
2. 貼到 ChatGPT（或等效模型），提醒模型嚴格按照百分比計算。
3. 將回傳 JSON 貼回儀表板；系統會驗證欄位與分數。
4. 理由欄位需包含：貼題/離題句、命中/缺漏要點、正確/錯誤陳述、支撐/未支撐句子與比例。
5. 儀表板會即時更新總分，並在「GPT分析」頁顯示改進建議。

## 常見問題

- **語義相似度顯示停用**：確定安裝 `sentence-transformers`、手動載入模型後重新啟動 Streamlit。
- **JSON 驗證失敗**：檢查是否是全形引號或缺少欄位；系統會提示缺漏欄位名稱。
- **需要詳細評分說明**：可閱讀「📝 GPT 補充說明」頁面，裡面整理了 prompt 設計原理與人工評審技巧。

## 版本控制

此儀表板仍以 `streamlit_dashboard_v2_with_manual_gpt.py` 為主體，其餘舊檔（例如 `streamlit_dashboard.py`、早期 README）僅供參考。請確保開啟、維護與部署皆使用這一份程式。

---
若要進一步自動化或擴充指標，可於 `generate_gpt_prompt()` 內調整 Prompt，或在 `rag_evaluation_two_models_v2.py` 擴充語義/關鍵詞演算法。
