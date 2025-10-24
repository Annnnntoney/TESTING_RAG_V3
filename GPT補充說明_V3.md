# 🧩 大型語言模型 (LLM) 輸出品質評估指標 v2

（依 AWS Bedrock Model Evaluation 原則修訂）

---

## 1️⃣ 相關性 (Relevance)

### 評分意義

衡量模型是否緊扣題目主軸，回答內容是否與問題語意一致、無偏題或重複。

### 參考來源與評估方式

- **AWS Bedrock – Relevance 指標**：由 LLM-as-a-Judge 依據評分提示（rubric）判定回答是否與問題緊密相關。分數通常正規化為 0–1 或 Likert 轉換。
- **FiRA (Fine-Grained Relevance Annotations, 2020)**：採細粒度標註，對句子或片段判斷「相關／不相關」，統計比例。
- **Sun & Wang (2024)**：提出句子層級語義貼合度指標，可用於語義相似度輔助。

### 實務評估流程（延伸／改作）

1. 將生成回答拆分為句子。
2. 為每句標註 On-Topic (1) 或 Off-Topic (0)。
3. 計算貼題比例：
   $$p = \frac{\text{On-Topic 句數}}{\text{總句數}}$$
4. 最終分數 = p × 100。
5. 若採語義相似度輔助，可將 cosine similarity ≧ 0.8 視為 On-Topic。

### 評分表（每 10% = 10 分）

| 貼題比例 p | 分數區間 | 等級說明 |
|-----------|---------|--------|
| p ≥ 0.90 | 90–100 | 幾乎所有句子皆貼題，主軸明確 |
| 0.80 ≤ p < 0.90 | 80–89 | 高度相關，僅少數句子偏離主題 |
| 0.70 ≤ p < 0.80 | 70–79 | 大多相關，部分離題 |
| 0.60 ≤ p < 0.70 | 60–69 | 過半相關，主軸仍可辨識 |
| 0.50 ≤ p < 0.60 | 50–59 | 僅部分相關，焦點模糊 |
| 0.40 ≤ p < 0.50 | 40–49 | 偏離主題為多，資訊混雜 |
| 0.30 ≤ p < 0.40 | 30–39 | 少數句子相關，主題偏差明顯 |
| 0.20 ≤ p < 0.30 | 20–29 | 幾乎未聚焦主題 |
| 0.10 ≤ p < 0.20 | 10–19 | 僅極少內容相關 |
| p < 0.10 | 0–9 | 完全離題 |

### 📚 主要參考來源

- AWS 官方文件：Model Evaluation Metrics – Relevance
- FiRA: Fine-Grained Relevance Annotations for Multi-Task Document Ranking and Question Answering (2020)
- Sun & Wang, 2024 – Sentence-Level Metrics Predicting Human Sentence Comprehension

---

## 2️⃣ 完整性 (Completeness)

### 評分意義

衡量回答是否涵蓋所有必要資訊要點，並具備足夠的內容深度與資訊整合性。

### 參考來源與評估方式

- **AWS Bedrock – Completeness 指標**：判斷模型是否回應所有問題面向，分數常由 judge 模型根據「覆蓋程度」打分。
- **AWS 自訂指標（Coverage + Depth）**：官方建議將覆蓋率與深度分開評估後再合併。
- **Tam et al. (2024, QUEST 框架)**：採列要點清單、標示命中與缺漏，再綜合深度與脈絡判斷。

### 🧮 評估方法（延伸／改作）

1. 列出必要要點：逐一標記 Covered／Partially／Missing。
2. Coverage 命中率 q：
   $$q = \frac{\text{Covered} + 0.5 \times \text{Partially}}{\text{總要點數}}$$
3. 品質係數 k：綜合以下三維度取平均（範圍 0.80–1.00）：
   - Depth（深度）
   - Context Utilization（上下文利用）
   - Information Synthesis（整合性）
4. 總分公式：
   $$\text{Score} = q \times 100 \times k$$
   
   若 q ≥ 0.90 但內容淺薄（僅羅列名詞），則強制 k ≤ 0.89（上限 89 分）。

### 🧾 綜合評分表（每 10% = 10 分）

| 綜合完整度（q × k） | 分數區間 | 等級說明 |
|------------------|---------|--------|
| ≥ 0.90 | 90–100 | 完整覆蓋、深度充分、資訊整合優秀；若僅羅列則上限 89。 |
| 0.80 ≤ (q × k) < 0.90 | 80–89 | 幾乎完整，少數細節略顯不足。 |
| 0.70 ≤ (q × k) < 0.80 | 70–79 | 大部分覆蓋，部分解釋略淺。 |
| 0.60 ≤ (q × k) < 0.70 | 60–69 | 覆蓋約 2/3，脈絡略有缺。 |
| 0.50 ≤ (q × k) < 0.60 | 50–59 | 僅部分要點完整，深度不足。 |
| 0.40 ≤ (q × k) < 0.50 | 40–49 | 覆蓋不足一半，內容片段化。 |
| 0.30 ≤ (q × k) < 0.40 | 30–39 | 少數要點被觸及，整體結構不足。 |
| 0.20 ≤ (q × k) < 0.30 | 20–29 | 僅零碎描述，缺乏關聯。 |
| 0.10 ≤ (q × k) < 0.20 | 10–19 | 幾乎未涵蓋必要資訊。 |
| (q × k) < 0.10 | 0–9 | 完全缺乏完整性與深度。 |

### 📈 建議使用方式

| 維度 | 意義 | 可量化方式 |
|------|------|---------|
| Coverage | 命中率 | 以 JIEBA 關鍵詞覆蓋率計算 q |
| Depth | 詳細程度 | 人工標註或詞數／句長近似 |
| Context Utilization | 是否使用題幹上下文 | 語義相似度 |
| Synthesis | 資訊整合能力 | LLM 判斷 0.8–1.0 分 |

### 📚 主要參考來源

- AWS Blog: Use custom metrics to evaluate your generative AI application with Amazon Bedrock
- AWS Docs: Model Evaluation Metrics – Completeness
- Tam et al., 2024 – QUEST Framework (PMC11437138)

---

## 3️⃣ 準確性 (Correctness / Accuracy)

### 評分意義

衡量回答的事實正確性與與標準答案一致程度。

### 參考來源與評估方式

- **AWS Bedrock – Correctness**：由 judge 模型依 rubric 判斷回答是否正確。
- **FActScore (Min et al., 2023)**：將回答拆成原子事實（atomic facts）逐條比對真值。

### 實務評估流程

1. 將回答拆解為原子事實。
2. 比對標準答案，標註 Correct／Incorrect。
3. 計算正確率：
   $$r = \frac{\text{正確事實數}}{\text{總事實數}}$$
4. 可輔以 F1-score 或語義相似度（cosine）。
5. 最終分數 = r × 100。

### 評分表（每 10% = 10 分）

| 正確率 r | 分數區間 | 等級說明 |
|---------|---------|--------|
| r ≥ 0.90 | 90–100 | 完全正確，無錯誤 |
| 0.80 ≤ r < 0.90 | 80–89 | 僅極少輕微錯誤 |
| 0.70 ≤ r < 0.80 | 70–79 | 多數正確，少數錯誤 |
| 0.60 ≤ r < 0.70 | 60–69 | 約 2/3 正確 |
| 0.50 ≤ r < 0.60 | 50–59 | 一半正確，一半錯誤 |
| 0.40 ≤ r < 0.50 | 40–49 | 錯誤比例明顯 |
| 0.30 ≤ r < 0.40 | 30–39 | 錯誤為多 |
| 0.20 ≤ r < 0.30 | 20–29 | 僅少數正確陳述 |
| 0.10 ≤ r < 0.20 | 10–19 | 幾乎全錯 |
| r < 0.10 | 0–9 | 完全錯誤或虛構內容 |

### 📚 主要參考來源

- AWS Docs: Model Evaluation Metrics – Correctness
- Min et al., 2023 – FActScore: Factual Consistency in LLMs

---

## 4️⃣ 忠誠度 (Faithfulness)

### 評分意義

衡量回答是否忠於原始檢索資料／上下文，避免虛構（hallucination）。

### 參考來源與評估方式

- **AWS Bedrock – Faithfulness**：檢查回答是否僅使用提供的 context，不加入外部內容。
- **Maynez et al. (2020)**：在摘要任務中逐句標註 Supported／Unsupported。
- **Fadeeva et al. (2025)**：針對 RAG 系統提出「Faithfulness-aware 檢核」與不確定性評估。

### 實務評估流程

1. 將回答拆為句子。
2. 每句標記為 Supported／Partially Supported／Unsupported。
3. 計算支撐比例：
   $$f = \frac{\text{Supported} + 0.5 \times \text{Partially}}{\text{總句數}}$$
4. 最終分數 = f × 100。
5. 若出現大量無依據斷言（hallucination），應落於 60 分以下區間。

### 評分表（每 10% = 10 分）

| 支撐比例 f | 分數區間 | 等級說明 |
|-----------|---------|--------|
| f ≥ 0.90 | 90–100 | 完全忠於來源，無虛構 |
| 0.80 ≤ f < 0.90 | 80–89 | 絕大多數有憑有據 |
| 0.70 ≤ f < 0.80 | 70–79 | 多數忠實，少量過度延伸 |
| 0.60 ≤ f < 0.70 | 60–69 | 約 2/3 具支撐，其餘存疑 |
| 0.50 ≤ f < 0.60 | 50–59 | 支撐不足，引用混雜 |
| 0.40 ≤ f < 0.50 | 40–49 | 多數內容未受支撐 |
| 0.30 ≤ f < 0.40 | 30–39 | 僅少數句子有支撐 |
| 0.20 ≤ f < 0.30 | 20–29 | 失真明顯 |
| 0.10 ≤ f < 0.20 | 10–19 | 幾乎全為未支撐斷言 |
| f < 0.10 | 0–9 | 完全虛構／無依據 |

### 📚 主要參考來源

- AWS Docs: Model Evaluation Metrics – Faithfulness
- Maynez et al., 2020 – On Faithfulness and Factuality in Abstractive Summarization
- Fadeeva et al., 2025 – Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of RAG

---

## ⚖️ 總體評分權重建議（非 RAG / RAG 任務）

| 指標 | 非 RAG 權重 | RAG 權重 |
|------|-----------|---------|
| Correctness | 0.45 | 0.35 |
| Completeness | 0.30 | 0.25 |
| Relevance | 0.25 | 0.15 |
| Faithfulness | — | 0.25 |

### 整體分數計算公式

$$\text{Overall Score} = \sum_i w_i \times \text{Score}_i$$

---

## 📝 使用說明

- 本評估框架可直接應用於 Notion、Google Sheets 或企業評估系統
- 建議根據具體任務需求調整權重係數
- 可搭配 LLM-as-a-Judge 自動化評分流程