"""
RAG 評估儀表板 v2.0 - 整合人工 GPT 評審
==========================================

特色功能：
1. 三層評估架構（關鍵詞、語義、GPT）
2. GPT Prompt 生成器（可直接複製到 ChatGPT）
3. GPT 回應貼上區（人工輸入評分）
4. 所有指標同時呈現、即時更新
5. 完整的資料輸入輸出支援

版本：2.0 with Manual GPT
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import os
import re
import ast
from pathlib import Path
from rag_evaluation_two_models_v2 import RAGEvaluatorV2
from evaluation_history_manager import EvaluationHistoryManager

# 設定頁面配置
st.set_page_config(
    page_title="RAG 評估儀表板 v2.0 - 整合人工 GPT 評審",
    #page_icon="🆚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session state
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'evaluator_instance' not in st.session_state:
    st.session_state.evaluator_instance = None
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'gpt_responses_original' not in st.session_state:
    st.session_state.gpt_responses_original = {}
if 'gpt_responses_optimized' not in st.session_state:
    st.session_state.gpt_responses_optimized = {}
if 'history_manager' not in st.session_state:
    st.session_state.history_manager = EvaluationHistoryManager()
if 'current_excel_filename' not in st.session_state:
    st.session_state.current_excel_filename = None
if 'gpt_responses_loaded' not in st.session_state:
    st.session_state.gpt_responses_loaded = False
if 'display_semantic_metric' not in st.session_state:
    st.session_state.display_semantic_metric = True
if 'gpt_selected_dimensions' not in st.session_state:
    st.session_state.gpt_selected_dimensions = ['relevance', 'completeness', 'accuracy', 'faithfulness']
if 'gpt_dimension_weights' not in st.session_state:
    st.session_state.gpt_dimension_weights = {
        'relevance': 0.25,
        'completeness': 0.25,
        'accuracy': 0.25,
        'faithfulness': 0.25,
    }

# 工具函數
def split_into_sentences(text: str):
    """將文字切成句子列表"""
    if not isinstance(text, str) or not text.strip():
        return []

    sentences = re.split(r'[。！？!?\n\r]+', text)
    return [s.strip() for s in sentences if s.strip()]


def compute_sentence_similarity(evaluator: RAGEvaluatorV2, sentences, answer: str):
    """計算每個句子與回答的語義相似度"""
    results = []
    if not evaluator or not evaluator.enable_semantic or not sentences:
        return results

    if answer is None or (isinstance(answer, float) and np.isnan(answer)) or str(answer).strip() == "":
        return [(sent, 0.0) for sent in sentences]

    for sent in sentences:
        try:
            score, _ = evaluator.calculate_semantic_similarity(sent, answer)
        except Exception:
            score = 0.0
        results.append((sent, score))

    return results


def format_reference_to_list(reference_text: str):
    """將參考內容拆成便於展示的條列"""
    if not isinstance(reference_text, str):
        return []

    lines = [line.strip() for line in reference_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        # 若只有單段，盡量依數字或頓號再拆分
        segments = re.split(r'\d+\.|[、；;]', reference_text)
        lines = [seg.strip() for seg in segments if seg.strip()]
    return lines


# GPT Prompt 生成函數
def generate_gpt_prompt(question, reference_keywords, answer, version="optimized", question_id=1):
    """生成 GPT 評審 prompt - 確保評分一致性"""
    prompt = f"""你是專業的RAG系統評估專家。請嚴格按照以下標準評估，確保評分一致性。

【問題 {question_id}】
{question}

【必須包含的關鍵資訊】
{reference_keywords}

【待評估回答（{version}版本）】
{answer}

【評分標準 ‑ 請嚴格遵守】

🎯 相關性 (Relevance) ─ 回答內容是否針對問題核心  
參考：Dr3: Ask Large Language Models Not to Give Off-Topic Answers (arXiv 2024)； 
1. 將回答拆成句子或條列，標記為 Strictly On-Topic / Off-Topic，並列出兩者。  
2. 計算貼題比例 p = 貼題句數 ÷ 總句數。  
3. 依下表給分，並在 reasoning 中用人類語氣說明：哪些句子貼題、哪些離題、為何加減分。  
   • 90‑100 分：p ≥ 0.90  
   • 80‑89 分：0.80 ≤ p < 0.90  
   • 70‑79 分：0.70 ≤ p < 0.80  
   • 60‑69 分：0.60 ≤ p < 0.70  
   • 50‑59 分：0.50 ≤ p < 0.60  
   • 40‑49 分：0.40 ≤ p < 0.50  
   • 30‑39 分：0.30 ≤ p < 0.40  
   • 20‑29 分：0.20 ≤ p < 0.30  
   • 10‑19 分：0.10 ≤ p < 0.20  
   • 0‑9 分：p < 0.10

📋 完整性 (Completeness) ─ 是否涵蓋所有必要資訊  
參考：AWS Bedrock “Information Comprehensiveness”[Coverage+Depth]；QUEST “Comprehensiveness”[80]；  
1. 列出【必須包含的關鍵資訊】的每個要點，並標記為 Covered / Partially / Missing。  
2. 命中率 q = (Covered + 0.5 × Partially) ÷ 總要點。  
3. 對每個 Covered 或 Partially 的要點，額外評估是否提供了充分細節、上下文或資料整合，並在 reasoning 中具體說明深度表現。  
4. 若整體內容雖覆蓋完整但說明明顯淺薄（僅名詞羅列、缺乏邏輯或補充），則整體分數上限為 89 分。  
5. 依下表給分；區間內可線性內插並四捨五入為整數。在 reasoning 中列出命中、部分命中、缺漏的要點及深度分析，最後用人類語氣總結整體影響。  

   • 90-100 分：q ≥ 0.90 且每項要點解釋充分（若深度不足，上限 89）  
   • 80-89 分：0.80 ≤ q < 0.90  
   • 70-79 分：0.70 ≤ q < 0.80  
   • 60-69 分：0.60 ≤ q < 0.70  
   • 50-59 分：0.50 ≤ q < 0.60  
   • 40-49 分：0.40 ≤ q < 0.50  
   • 30-39 分：0.30 ≤ q < 0.40  
   • 20-29 分：0.20 ≤ q < 0.30  
   • 10-19 分：0.10 ≤ q < 0.20  
   • 0-9 分：q < 0.10

✅ 準確性 (Accuracy) ─ 資訊是否正確  
參考：Min et al., 2023 “FactScore”；Lee et al., 2023 “RLAIF vs. RLHF” 附錄事實檢查； 
1. 列出回答中可驗證的陳述 S1…Sn，標記為 Correct / Incorrect / Unverifiable。  
2. 正確率 r = Correct ÷ (Correct + Incorrect)。  
3. 依下表給分，在 reasoning 中列出正確與錯誤的陳述、指出主要錯誤來源，並用人類語氣總結。  
   • 90‑100 分：r ≥ 0.95  
   • 80‑89 分：0.85 ≤ r < 0.95  
   • 70‑79 分：0.75 ≤ r < 0.85  
   • 60‑69 分：0.65 ≤ r < 0.75  
   • 50‑59 分：0.55 ≤ r < 0.65  
   • 40‑49 分：0.45 ≤ r < 0.55  
   • 30‑39 分：0.35 ≤ r < 0.45  
   • 20‑29 分：0.25 ≤ r < 0.35  
   • 10‑19 分：0.15 ≤ r < 0.25  
   • 0‑9 分：r < 0.15

🔒 忠誠度 (Faithfulness) ─ 是否基於原始資料  
參考：Maynez et al., 2020 “On Faithfulness…”；Lee et al., 2021 “Evaluation of RAG Metrics…”；  
1. 對每句或每個陳述標記為 Supported / Partially Supported / Unsupported，並列出各類句子。  
2. 支撐比例 f = (Supported + 0.5 × Partially) ÷ 全部陳述。  
3. 依下表給分，在 reasoning 中清楚說明：哪些句子有來源、哪些缺乏依據或屬合理推論，最後以人類語氣總結。  
   • 90‑100 分：f ≥ 0.90  
   • 80‑89 分：0.80 ≤ f < 0.90  
   • 70‑79 分：0.70 ≤ f < 0.80  
   • 60‑69 分：0.60 ≤ f < 0.70  
   • 50‑59 分：0.50 ≤ f < 0.60  
   • 40‑49 分：0.40 ≤ f < 0.50  
   • 30‑39 分：0.30 ≤ f < 0.40  
   • 20‑29 分：0.20 ≤ f < 0.30  
   • 10‑19 分：0.10 ≤ f < 0.20  
   • 0‑9 分：f < 0.10


【評估流程 - 請按順序執行】
1. 先仔細閱讀問題，理解問題要求。
2. 數出【必須包含的關鍵資訊】中的關鍵概念總數。
3. 逐句分析回答內容，為四個維度準備對應的句子清單與比例。
4. 依據上方四個評分表換算各維度分數，並確認是否落在正確的百分比區間。
5. 在 reasoning 中以人類語氣整理：列出貼題/離題、命中/缺漏、正確/錯誤、Supported/Unsupported 的句子與比例，最後用一兩句話總結加減分原因。

請務必按照以下JSON格式回傳：
{{
  "question_id": {question_id},
  "relevance": <分數>,
  "relevance_reasoning": "貼題句：[...]｜離題句：[...]｜結論：p=Y%，因此給X分，原因是...",
  "completeness": <分數>,
  "completeness_reasoning": "命中要點：[...]｜缺漏/部分要點：[...]｜結論：命中率Z%(Y/X)，因此給X分，原因是...",
  "accuracy": <分數>,
  "accuracy_reasoning": "正確陳述：[...]｜錯誤/不可驗證：[...]｜結論：正確率Z%(Y/X)，因此給X分，指出主要錯誤...",
  "faithfulness": <分數>,
  "faithfulness_reasoning": "Supported：[...]｜Unsupported/推論：[...]｜結論：支撐率Z%(Y/X)，因此給X分，說明缺乏來源的句子...",
  "overall": <四個維度平均分>,
  "overall_reasoning": "四個維度平均：(X+Y+Z+W)/4=總分"
}}

重要提醒：請確保每次評估都嚴格按照百分比計算，不要依賴主觀感覺。每個reasoning必須包含具體數據。"""

    return prompt

# 解析 GPT 回應
def normalize_json_like_text(text: str) -> str:
    """將常見的全形/彎引號替換成標準 ASCII 字元，方便 JSON 解析"""
    if not isinstance(text, str):
        return text

    normalized = text

    # 先處理鍵名：將「“key” :」轉換為標準 JSON 格式
    normalized = re.sub(r'“([^”]+)”\s*:', lambda m: f'"{m.group(1)}":', normalized)

    # 處理以全形引號包裹的值，確保起訖使用標準雙引號
    normalized = re.sub(r':\s*“', ': "', normalized)
    normalized = re.sub(r'”(?=\s*[,\n}])', '"', normalized)

    replacements = {
        '“': "'",
        '”': "'",
        '＂': '"',
        '「': "'",
        '」': "'",
        '『': "'",
        '』': "'",
        '‘': "'",
        '’': "'",
        '＇': "'",
    }

    for src, target in replacements.items():
        normalized = normalized.replace(src, target)

    return normalized


def parse_gpt_response(response_text):
    """解析 ChatGPT 的 JSON 回應，並容錯處理常見的格式問題"""

    if not response_text:
        return {"error": "回應為空白", "raw_response": response_text}

    candidates = []
    raw_text = response_text.strip()
    candidates.append(raw_text)

    normalized_text = normalize_json_like_text(raw_text)
    if normalized_text != raw_text:
        candidates.append(normalized_text)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                literal_result = ast.literal_eval(candidate)
                if isinstance(literal_result, dict):
                    return literal_result
            except (ValueError, SyntaxError):
                pass
            json_match = re.search(r'\{.*\}', candidate, re.DOTALL)
            if json_match:
                json_snippet = json_match.group().strip()
                try:
                    return json.loads(json_snippet)
                except json.JSONDecodeError:
                    try:
                        literal_result = ast.literal_eval(json_snippet)
                        if isinstance(literal_result, dict):
                            return literal_result
                    except (ValueError, SyntaxError):
                        continue

    return {"error": "無法解析 GPT 回應", "raw_response": response_text}


GPT_DIMENSION_LABELS = {
    'relevance': '🎯 相關性',
    'completeness': '📋 完整性',
    'accuracy': '✅ 準確性',
    'faithfulness': '🔒 忠誠度',
}

DEFAULT_GPT_DIMENSIONS = list(GPT_DIMENSION_LABELS.keys())


def get_selected_gpt_dimensions() -> list:
    """取得目前選擇的 GPT 綜合評分維度（至少回傳一個）。"""
    selected = st.session_state.get('gpt_selected_dimensions', DEFAULT_GPT_DIMENSIONS)
    if not selected:
        return DEFAULT_GPT_DIMENSIONS

    filtered = [dim for dim in selected if dim in GPT_DIMENSION_LABELS]
    return filtered or DEFAULT_GPT_DIMENSIONS


def get_gpt_dimension_weights(selected_dims: list) -> dict:
    """以目前選擇的維度回傳歸一化權重，預設平均。"""
    weights_setting = st.session_state.get('gpt_dimension_weights', {})
    weights = {}
    total = 0.0
    for dim in selected_dims:
        value = weights_setting.get(dim)
        if isinstance(value, (int, float)) and value >= 0:
            weights[dim] = float(value)
            total += float(value)
        else:
            weights[dim] = 0.0

    if total <= 0:
        # 若全部為 0 或無設定，改採平均權重
        equal_weight = 1.0 / len(selected_dims)
        return {dim: equal_weight for dim in selected_dims}

    return {dim: value / total for dim, value in weights.items()}


def compute_gpt_overall(gpt_data: dict, selected_dims: list | None = None, dim_weights: dict | None = None) -> float:
    """依照選取的維度重新計算 GPT 綜合評分。"""
    if not isinstance(gpt_data, dict):
        return 0.0

    dimensions = selected_dims or get_selected_gpt_dimensions()
    weights = dim_weights or get_gpt_dimension_weights(dimensions)

    score_sum = 0.0
    applied_weight = 0.0
    for dim in dimensions:
        raw_value = gpt_data.get(dim)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        weight = weights.get(dim, 0.0)
        score_sum += value * weight
        applied_weight += weight

    if applied_weight > 0:
        return score_sum / applied_weight

    fallback = gpt_data.get('overall')
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return 0.0


def format_gpt_weight_summary(selected_dims: list, dim_weights: dict | None = None) -> str:
    weights = dim_weights or get_gpt_dimension_weights(selected_dims)
    parts = []
    for dim in selected_dims:
        label = GPT_DIMENSION_LABELS.get(dim, dim)
        parts.append(f"{label} {weights.get(dim, 0)*100:.0f}%")
    return '、'.join(parts)

# 驗證評分一致性函數
def validate_scoring_consistency(parsed_response, question_text, answer_text):
    """驗證評分的邏輯一致性和完整性"""
    
    warnings = []
    errors = []
    
    # 檢查必要欄位
    required_fields = [
        'relevance', 'relevance_reasoning',
        'completeness', 'completeness_reasoning', 
        'accuracy', 'accuracy_reasoning',
        'faithfulness', 'faithfulness_reasoning',
        'overall', 'overall_reasoning'
    ]
    
    for field in required_fields:
        if field not in parsed_response:
            errors.append(f"缺少必要欄位: {field}")
            continue

        value = parsed_response.get(field)
        if value is None:
            errors.append(f"缺少必要欄位: {field}")
            continue

        if isinstance(value, str) and not value.strip():
            errors.append(f"缺少必要欄位: {field}")
    
    # 如果有錯誤，直接返回
    if errors:
        return warnings, errors
    
    # 檢查分數範圍
    score_fields = ['relevance', 'completeness', 'accuracy', 'faithfulness', 'overall']
    for score_field in score_fields:
        score = parsed_response.get(score_field, 0)
        if not isinstance(score, (int, float)) or not (0 <= score <= 100):
            errors.append(f"{score_field} 分數必須在 0-100 之間，目前為: {score}")
    
    # 檢查overall是否為四個維度的平均值
    if not errors:  # 只有在沒有錯誤時才檢查
        scores = {
            'relevance': parsed_response.get('relevance', 0),
            'completeness': parsed_response.get('completeness', 0), 
            'accuracy': parsed_response.get('accuracy', 0),
            'faithfulness': parsed_response.get('faithfulness', 0)
        }
        
        expected_overall = sum(scores.values()) / 4
        actual_overall = parsed_response.get('overall', 0)
        
        if abs(expected_overall - actual_overall) > 2:  # 容許2分誤差
            warnings.append(f"總分可能不一致：期望{expected_overall:.1f}，實際{actual_overall}")
    
    # 檢查reasoning是否包含具體數據
    reasoning_fields = ['relevance_reasoning', 'completeness_reasoning', 'accuracy_reasoning', 'faithfulness_reasoning']
    for reasoning_field in reasoning_fields:
        reasoning = parsed_response.get(reasoning_field, '')
        if '%' not in reasoning and '個' not in reasoning and '分析過程' not in reasoning:
            warnings.append(f"{reasoning_field}缺少具體數據或分析過程說明")
    
    return warnings, errors

# 自動保存評估結果到歷史紀錄
def auto_save_evaluation(actual_question_id, results_df, weights, selected_dims=None, dim_weights=None):
    """
    自動保存評估結果到歷史紀錄

    策略：
    1. 如果兩個版本都有 GPT 評分 → 保存完整評估
    2. 如果只有一個版本有 GPT 評分 → 保存該版本（另一版本用 0 填充）

    Args:
        actual_question_id: 實際的問題序號（來自 '序號' 欄位）
        results_df: 評估結果 DataFrame
        weights: 權重設定
    """
    # 檢查是否至少有一個版本有 GPT 評分（使用實際序號）
    has_original = actual_question_id in st.session_state.gpt_responses_original
    has_optimized = actual_question_id in st.session_state.gpt_responses_optimized

    if not (has_original or has_optimized):
        return False  # 兩個版本都沒有 GPT 評分，不保存

    try:
        # 在 DataFrame 中查找對應的行（使用 '序號' 欄位匹配）
        matching_rows = results_df[results_df['序號'] == actual_question_id]
        if matching_rows.empty:
            print(f"⚠️ 找不到序號 {actual_question_id} 的問題")
            return False

        row = matching_rows.iloc[0]

        # 準備原始版本評分
        selected_dims = selected_dims or get_selected_gpt_dimensions()
        dim_weights = dim_weights or get_gpt_dimension_weights(selected_dims)

        if has_original:
            gpt_orig = st.session_state.gpt_responses_original[actual_question_id]
            original_scores = {
                "keyword_score": row.get('KEYWORD_COVERAGE_ORIGINAL', 0),
                "semantic_score": row.get('SEMANTIC_SIMILARITY_ORIGINAL', 0),
                "gpt_relevance": gpt_orig.get('relevance', 0),
                "gpt_completeness": gpt_orig.get('completeness', 0),
                "gpt_accuracy": gpt_orig.get('accuracy', 0),
                "gpt_faithfulness": gpt_orig.get('faithfulness', 0),
                "gpt_overall": compute_gpt_overall(gpt_orig, selected_dims, dim_weights),
                "gpt_reasoning": gpt_orig.get('reasoning', ''),
                "final_score": row.get('FINAL_SCORE_ORIGINAL', 0)
            }
        else:
            # 原始版本沒有 GPT 評分，只保存關鍵詞和語義
            original_scores = {
                "keyword_score": row.get('KEYWORD_COVERAGE_ORIGINAL', 0),
                "semantic_score": row.get('SEMANTIC_SIMILARITY_ORIGINAL', 0),
                "gpt_relevance": 0,
                "gpt_completeness": 0,
                "gpt_accuracy": 0,
                "gpt_faithfulness": 0,
                "gpt_overall": 0,
                "gpt_reasoning": "",
                "final_score": row.get('FINAL_SCORE_ORIGINAL', 0)
            }

        # 準備優化版本評分
        if has_optimized:
            gpt_opt = st.session_state.gpt_responses_optimized[actual_question_id]
            optimized_scores = {
                "keyword_score": row.get('KEYWORD_COVERAGE_OPTIMIZED', 0),
                "semantic_score": row.get('SEMANTIC_SIMILARITY_OPTIMIZED', 0),
                "gpt_relevance": gpt_opt.get('relevance', 0),
                "gpt_completeness": gpt_opt.get('completeness', 0),
                "gpt_accuracy": gpt_opt.get('accuracy', 0),
                "gpt_faithfulness": gpt_opt.get('faithfulness', 0),
                "gpt_overall": compute_gpt_overall(gpt_opt, selected_dims, dim_weights),
                "gpt_reasoning": gpt_opt.get('reasoning', ''),
                "final_score": row.get('FINAL_SCORE_OPTIMIZED', 0)
            }
        else:
            # 優化版本沒有 GPT 評分，只保存關鍵詞和語義
            optimized_scores = {
                "keyword_score": row.get('KEYWORD_COVERAGE_OPTIMIZED', 0),
                "semantic_score": row.get('SEMANTIC_SIMILARITY_OPTIMIZED', 0),
                "gpt_relevance": 0,
                "gpt_completeness": 0,
                "gpt_accuracy": 0,
                "gpt_faithfulness": 0,
                "gpt_overall": 0,
                "gpt_reasoning": "",
                "final_score": row.get('FINAL_SCORE_OPTIMIZED', 0)
            }

        # 保存到歷史紀錄（使用實際序號）
        success = st.session_state.history_manager.save_evaluation(
            excel_filename=st.session_state.current_excel_filename,
            question_id=actual_question_id,
            question_text=row.get('測試問題', ''),
            reference_keywords=row.get('應回答之詞彙', ''),
            original_answer=row.get('ANSWER_ORIGINAL', ''),
            optimized_answer=row.get('ANSWER_OPTIMIZED', ''),
            original_scores=original_scores,
            optimized_scores=optimized_scores,
            weights=weights,
            metadata={
                "evaluation_date": datetime.now().isoformat(),
                "improvement": optimized_scores['final_score'] - original_scores['final_score'],
                "has_original_gpt": has_original,
                "has_optimized_gpt": has_optimized
            }
        )

        return success

    except Exception as e:
        print(f"❌ 自動保存失敗: {e}")
        return False

# 從歷史紀錄載入 GPT 評分
def load_gpt_from_history(excel_filename):
    """從歷史紀錄載入該檔案的 GPT 評分"""
    if not excel_filename:
        return

    try:
        evaluations = st.session_state.history_manager.get_evaluations_by_file(excel_filename)

        for eval_record in evaluations:
            # 使用實際 question_id（不需要轉換，直接使用原始序號）
            question_id = eval_record.get("question_id", 0)

            # 載入原始版本 GPT 評分
            original_scores = eval_record.get("scores", {}).get("original", {})
            if original_scores.get("gpt_overall", 0) > 0:
                st.session_state.gpt_responses_original[question_id] = {
                    "relevance": original_scores.get("gpt_relevance", 0),
                    "completeness": original_scores.get("gpt_completeness", 0),
                    "accuracy": original_scores.get("gpt_accuracy", 0),
                    "faithfulness": original_scores.get("gpt_faithfulness", 0),
                    "overall": original_scores.get("gpt_overall", 0),
                    "reasoning": original_scores.get("gpt_reasoning", "")
                }

            # 載入優化版本 GPT 評分
            optimized_scores = eval_record.get("scores", {}).get("optimized", {})
            if optimized_scores.get("gpt_overall", 0) > 0:
                st.session_state.gpt_responses_optimized[question_id] = {
                    "relevance": optimized_scores.get("gpt_relevance", 0),
                    "completeness": optimized_scores.get("gpt_completeness", 0),
                    "accuracy": optimized_scores.get("gpt_accuracy", 0),
                    "faithfulness": optimized_scores.get("gpt_faithfulness", 0),
                    "overall": optimized_scores.get("gpt_overall", 0),
                    "reasoning": optimized_scores.get("gpt_reasoning", "")
                }

        if evaluations:
            print(f"✅ 從歷史紀錄載入了 {len(evaluations)} 筆 GPT 評分")
            return len(evaluations)

    except Exception as e:
        print(f"⚠️ 載入歷史 GPT 評分失敗: {e}")

    return 0

# 標題和說明
st.title("🆚 RAG 評估儀表板 v2.0")
st.markdown("### RAG評估架構：關鍵詞 + 語義相似度 + GPT 人工評審")

# 側邊欄配置
with st.sidebar:
    st.header("📁 設定與檔案選擇")

    # 檔案選擇方式
    file_source = st.radio(
        "選擇檔案來源",
        ["📂 本地資料夾", "📤 上傳檔案"],
        help="選擇要從本地資料夾載入或上傳新檔案"
    )

    selected_file_path = None
    uploaded_file = None

    if file_source == "📂 本地資料夾":
        data_folder = "test_data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        st.caption(f"資料夾路徑：{data_folder}")

        try:
            all_files = os.listdir(data_folder)
            excel_files = [f for f in all_files
                          if f.endswith(('.xlsx', '.xls', '.csv')) and not f.startswith('~') and not f.startswith('.')]
        except Exception as e:
            st.error(f"讀取資料夾時發生錯誤：{str(e)}")
            excel_files = []

        if excel_files:
            selected_file = st.selectbox(
                "選擇要評估的檔案",
                excel_files,
                help="從 test_data 資料夾中選擇檔案"
            )
            selected_file_path = os.path.join(data_folder, selected_file)
            uploaded_file = selected_file_path

            file_info = os.stat(selected_file_path)
            st.info(f"檔案大小：{file_info.st_size / 1024:.1f} KB")
            st.success(f"✅ 已載入: {selected_file}")
        else:
            st.warning("⚠️ test_data 資料夾中沒有找到 Excel 或 CSV 檔案")

    else:  # 上傳檔案
        uploaded_file = st.file_uploader(
            "上傳測試結果Excel/CSV檔案",
            type=['xlsx', 'xls', 'csv'],
            help="請上傳包含向量知識庫(原始版)和智慧文檔知識庫(彙整版)回答的測試結果"
        )

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            selected_file_path = f"temp_uploaded.{file_extension}"
            with open(selected_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ 已載入: {uploaded_file.name}")

    # 知識庫選擇
    st.markdown("### 📚 知識庫設定")
    col1, col2 = st.columns(2)

    with col1:
        original_kb = st.selectbox(
            "原始版本",
            ["向量知識庫", "智慧文檔知識庫"],
            index=0,
            help="選擇原始版本使用的知識庫技術"
        )

    with col2:
        optimized_kb = st.selectbox(
            "優化版本",
            ["智慧文檔知識庫", "向量知識庫+優化"],
            index=0,
            help="選擇優化版本使用的知識庫技術"
        )

    # 評估層級設定
    st.markdown("### 🎯 評估層級設定")
    st.info("🔍 選擇要啟用的評估層級")

    enable_semantic = st.checkbox(
        "啟用語義相似度評估",
        value=True,
        help="使用 Sentence Transformers 計算語義相似度（推薦）"
    )

    enable_manual_gpt = st.checkbox(
        "啟用 GPT 人工評審",
        value=True,
        help="生成 GPT prompts，您可複製到 ChatGPT 並貼回評分（完全免費，無需 API）"
    )

    if enable_manual_gpt:
        st.success("✅ GPT 人工評審模式")
        st.info("💡 系統會生成 Prompt，請直接複製到 ChatGPT 進行分析")

    show_semantic_overview = st.checkbox(
        "概覽顯示語義相似度",
        value=st.session_state.display_semantic_metric,
        help="僅影響評估總覽指標卡片；即使不顯示，語義分數仍會計算並能在其他分頁使用。"
    )
    st.session_state.display_semantic_metric = show_semantic_overview

    # 評分權重設定
    st.markdown("### ⚖️ 評分權重設定")

    if enable_semantic and enable_manual_gpt:
        st.info("三種評估模式")
        weight_keyword = st.slider("關鍵詞權重", 0.0, 1.0, 0.3, 0.1)
        weight_semantic = st.slider("語義權重", 0.0, 1.0, 0.3, 0.1)
        weight_gpt = 1.0 - weight_keyword - weight_semantic
        st.metric("GPT 權重", f"{weight_gpt:.1f}")
    elif enable_semantic:
        st.info("雙層評估模式（關鍵詞 + 語義）")
        weight_keyword = st.slider("關鍵詞權重", 0.0, 1.0, 0.5, 0.1)
        weight_semantic = 1.0 - weight_keyword
        weight_gpt = 0.0
        st.metric("語義權重", f"{weight_semantic:.1f}")
    elif enable_manual_gpt:
        st.info("雙層評估模式（關鍵詞 + GPT）")
        weight_keyword = st.slider("關鍵詞權重", 0.0, 1.0, 0.4, 0.1)
        weight_gpt = 1.0 - weight_keyword
        weight_semantic = 0.0
        st.metric("GPT 權重", f"{weight_gpt:.1f}")
    else:
        st.info("��層評估模式（僅關鍵詞）")
        weight_keyword = 1.0
        weight_semantic = 0.0
        weight_gpt = 0.0

    weights = {
        "keyword": weight_keyword,
        "semantic": weight_semantic,
        "gpt": weight_gpt
    }

    # 顯著改善閾值
    st.markdown("### 🎯 分析設定")
    improvement_threshold = st.slider(
        "顯著改善閾值 (%)",
        min_value=5,
        max_value=50,
        value=10,
        help="當改善幅度超過此閾值時，標記為顯著改善"
    )

    # 歷史紀錄管理
    st.markdown("### 📚 歷史紀錄")
    all_evaluations = st.session_state.history_manager.get_all_evaluations()
    stats = st.session_state.history_manager.get_statistics()

    st.metric("累計評估題數", stats["total_evaluations"])
    st.metric("已評估檔案數", stats["files_evaluated"])

    if stats["total_evaluations"] > 0:
        st.metric(
            "平均改善幅度",
            f"{stats['avg_improvement']:.1f}",
            delta=f"{stats['avg_improvement']:.1f}%"
        )

    # 匯出歷史紀錄按鈕
    if st.button("📥 匯出完整歷史紀錄", use_container_width=True):
        output_path = f"evaluation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        if st.session_state.history_manager.export_to_excel(output_path):
            st.success(f"✅ 已匯出到 {output_path}")
        else:
            st.error("❌ 匯出失敗")

    # 清除歷史紀錄按鈕
    if st.button("🗑️ 清除歷史紀錄", type="secondary", use_container_width=True):
        if st.session_state.history_manager.clear_history():
            st.success("✅ 歷史紀錄已清除")
            st.rerun()
        else:
            st.error("❌ 清除失敗")

# 主要內容區
if uploaded_file is not None:
    # 處理檔案
    if isinstance(uploaded_file, str):
        temp_file_path = uploaded_file
        # 設定當前檔案名稱（用於歷史紀錄）
        st.session_state.current_excel_filename = os.path.basename(uploaded_file)
    else:
        temp_file_path = "temp_comparison_file.xlsx"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # 設定當前檔案名稱（用於歷史紀錄）
        st.session_state.current_excel_filename = uploaded_file.name

    # 根據選擇的知識庫類型建立評估器
    try:
        if original_kb == "向量知識庫" and optimized_kb == "智慧文檔知識庫":
            model_type = "cross"
        elif original_kb == "向量知識庫":
            model_type = "vector"
        else:
            model_type = "smart_doc"

        evaluator = RAGEvaluatorV2(
            temp_file_path,
            model_type=model_type,
            enable_semantic=enable_semantic,
            enable_gpt=False,  # 我們使用人工評審，不使用 API
            weights=weights
        )

        # 如果語義相似度在載入階段被停用，提醒使用者並同步狀態
        if enable_semantic and not evaluator.enable_semantic:
            st.warning("⚠️ 語義相似度模型未啟動，請確認已安裝 sentence-transformers 與 torch 套件。")

        enable_semantic = evaluator.enable_semantic
        weights = evaluator.weights

        st.session_state.evaluator_instance = evaluator

        # 執行評估（不包含 GPT，GPT 由人工提供）
        # 檢查是否已經有評估結果（避免重複評估導致語義相似度丟失）
        if st.session_state.comparison_results is None:
            with st.spinner("🔄 正在進行評估分析..."):
                results_df = evaluator.evaluate_all()
                st.session_state.comparison_results = results_df
        else:
            # 已經有評估結果，直接使用
            results_df = st.session_state.comparison_results

        # 從歷史紀錄載入 GPT 評分（如果有的話）
        if not st.session_state.gpt_responses_loaded and st.session_state.current_excel_filename:
            loaded_count = load_gpt_from_history(st.session_state.current_excel_filename)
            if loaded_count > 0:
                st.success(f"✅ 從歷史紀錄恢復了 {loaded_count} 筆 GPT 評分")
            st.session_state.gpt_responses_loaded = True

        # 清理臨時檔案
        if os.path.exists(temp_file_path) and not isinstance(uploaded_file, str):
            os.remove(temp_file_path)

    except Exception as e:
        st.error(f"❌ 評估過程中發生錯誤：{str(e)}")
        if os.path.exists(temp_file_path) and not isinstance(uploaded_file, str):
            os.remove(temp_file_path)
        st.stop()

    # 建立頁籤
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "📊 評估總覽",
            "🤖 GPT 人工評審",
            "📈 綜合比較圖表",
            "🔤 語義分析",
            "💬 GPT分析",
            "🎯 關鍵詞分析",
            "📥 下載結果",
            "📝 GPT 補充說明"
        ]
    )

    with tab1:
        st.markdown("### 📊 評估總覽")

        # 計算包含 GPT 的綜合評分
        results_df = st.session_state.comparison_results.copy()

        selected_gpt_dims = get_selected_gpt_dimensions()
        selected_gpt_weights = get_gpt_dimension_weights(selected_gpt_dims)
        selected_weight_summary = format_gpt_weight_summary(selected_gpt_dims, selected_gpt_weights)

        # 加入 GPT 評分（如果有）- 使用實際序號而非 DataFrame index
        for idx in range(len(results_df)):
            # 取得該行的實際序號
            actual_q_id = int(results_df.iloc[idx]['序號'])

            if actual_q_id in st.session_state.gpt_responses_original:
                gpt_data = st.session_state.gpt_responses_original[actual_q_id]
                results_df.at[idx, 'GPT_OVERALL_ORIGINAL'] = compute_gpt_overall(
                    gpt_data, selected_gpt_dims, selected_gpt_weights
                )
                results_df.at[idx, 'GPT_OVERALL_ORIGINAL_RAW'] = gpt_data.get('overall', 0)
            else:
                results_df.at[idx, 'GPT_OVERALL_ORIGINAL'] = 0
                results_df.at[idx, 'GPT_OVERALL_ORIGINAL_RAW'] = 0

            if actual_q_id in st.session_state.gpt_responses_optimized:
                gpt_data = st.session_state.gpt_responses_optimized[actual_q_id]
                results_df.at[idx, 'GPT_OVERALL_OPTIMIZED'] = compute_gpt_overall(
                    gpt_data, selected_gpt_dims, selected_gpt_weights
                )
                results_df.at[idx, 'GPT_OVERALL_OPTIMIZED_RAW'] = gpt_data.get('overall', 0)
            else:
                results_df.at[idx, 'GPT_OVERALL_OPTIMIZED'] = 0
                results_df.at[idx, 'GPT_OVERALL_OPTIMIZED_RAW'] = 0

        # 重新計算綜合評分（包含 GPT）
        # 注意：不覆蓋原始 results_df，保留原始的語義相似度分數
        results_df['FINAL_SCORE_ORIGINAL'] = (
            results_df['KEYWORD_COVERAGE_ORIGINAL'] * weights['keyword'] +
            results_df['SEMANTIC_SIMILARITY_ORIGINAL'] * weights['semantic'] +
            results_df['GPT_OVERALL_ORIGINAL'] * weights['gpt']
        )

        results_df['FINAL_SCORE_OPTIMIZED'] = (
            results_df['KEYWORD_COVERAGE_OPTIMIZED'] * weights['keyword'] +
            results_df['SEMANTIC_SIMILARITY_OPTIMIZED'] * weights['semantic'] +
            results_df['GPT_OVERALL_OPTIMIZED'] * weights['gpt']
        )

        results_df['FINAL_IMPROVEMENT'] = (
            results_df['FINAL_SCORE_OPTIMIZED'] - results_df['FINAL_SCORE_ORIGINAL']
        )

        # ⚠️ 不要覆蓋 session_state！保留原始評估數據
        # st.session_state.comparison_results = results_df  # <-- 移除這行

        # 關鍵指標卡片
        overview_blocks = []

        def render_overall():
            st.markdown("**📈 綜合評分**")
            avg_orig = results_df['FINAL_SCORE_ORIGINAL'].mean()
            avg_opt = results_df['FINAL_SCORE_OPTIMIZED'].mean()
            improvement = avg_opt - avg_orig
            color = '#28a745' if improvement > 0 else '#dc3545'
            st.markdown(f"<h1 style='color: {color}; margin: 0;'>{avg_opt:.1f}分</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if improvement > 0 else '↓'} {abs(improvement):.1f}分</p>", unsafe_allow_html=True)
            st.caption(
                "依照目前權重 (關鍵詞 {keyword:.0%} / 語義 {semantic:.0%} / GPT {gpt:.0%})"
                " 對每題的三層分數做加權平均後，再取所有題目的平均分。".format(
                    keyword=weights['keyword'],
                    semantic=weights['semantic'],
                    gpt=weights['gpt']
                )
            )

        def render_keyword():
            st.markdown("**🎯 關鍵詞覆蓋率**")
            keyword_improvement = results_df['KEYWORD_COVERAGE_OPTIMIZED'].mean() - results_df['KEYWORD_COVERAGE_ORIGINAL'].mean()
            color = '#28a745' if keyword_improvement > 0 else '#dc3545'
            st.markdown(f"<h1 style='color: {color}; margin: 0;'>{results_df['KEYWORD_COVERAGE_OPTIMIZED'].mean():.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if keyword_improvement > 0 else '↓'} {abs(keyword_improvement):.1f}%</p>", unsafe_allow_html=True)
            st.caption("自動比對回答與『應回答之詞彙』的命中比例，平均所有題目後取得此數值。")

        def render_semantic():
            if enable_semantic:
                st.markdown("**🔤 語義相似度**")
                semantic_improvement = results_df['SEMANTIC_SIMILARITY_OPTIMIZED'].mean() - results_df['SEMANTIC_SIMILARITY_ORIGINAL'].mean()
                color = '#28a745' if semantic_improvement > 0 else '#dc3545'
                st.markdown(f"<h1 style='color: {color}; margin: 0;'>{results_df['SEMANTIC_SIMILARITY_OPTIMIZED'].mean():.1f}%</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if semantic_improvement > 0 else '↓'} {abs(semantic_improvement):.1f}%</p>", unsafe_allow_html=True)
                st.caption("使用 Sentence-Transformers 量測『應回答內容』與實際回答的向量餘弦相似度，取所有題目平均值。")
            else:
                st.info("語義相似度未啟用")

        def render_gpt():
            if enable_manual_gpt:
                st.markdown("**🤖 GPT 評分**")

                gpt_original_ids = set(st.session_state.gpt_responses_original.keys())
                gpt_optimized_ids = set(st.session_state.gpt_responses_optimized.keys())
                evaluated_question_ids = gpt_original_ids | gpt_optimized_ids
                total_count = len(results_df)
                selected_dims = selected_gpt_dims
                selected_summary_local = selected_weight_summary

                if evaluated_question_ids:
                    seq_series = results_df['序號'].astype(int)

                    mask_opt = seq_series.isin(gpt_optimized_ids)
                    mask_orig = seq_series.isin(gpt_original_ids)
                    mask_both = seq_series.isin(gpt_original_ids & gpt_optimized_ids)

                    avg_opt = results_df.loc[mask_opt, 'GPT_OVERALL_OPTIMIZED'].mean() if mask_opt.any() else None
                    avg_orig = results_df.loc[mask_orig, 'GPT_OVERALL_ORIGINAL'].mean() if mask_orig.any() else None

                    display_score = avg_opt if avg_opt is not None else avg_orig
                    if display_score is not None:
                        color = '#2196F3'
                        delta_html = ""

                        if mask_both.any():
                            improvement = (
                                results_df.loc[mask_both, 'GPT_OVERALL_OPTIMIZED'].mean() -
                                results_df.loc[mask_both, 'GPT_OVERALL_ORIGINAL'].mean()
                            )
                            if improvement > 0:
                                color = '#28a745'
                                delta_html = f"<p style='color: {color}; font-size: 18px;'>↑ {abs(improvement):.1f}分</p>"
                            elif improvement < 0:
                                color = '#dc3545'
                                delta_html = f"<p style='color: {color}; font-size: 18px;'>↓ {abs(improvement):.1f}分</p>"
                            else:
                                delta_html = f"<p style='color: {color}; font-size: 18px;'>→ 0.0分</p>"

                        st.markdown(
                            f"<h1 style='color: {color}; margin: 0;'>{display_score:.1f}分</h1>",
                            unsafe_allow_html=True
                        )
                        if delta_html:
                            st.markdown(delta_html, unsafe_allow_html=True)
                        st.caption(
                            "人工 GPT 評審依照{summary}的加權平均；若兩版本皆完成評審會顯示改進幅度。".format(
                                summary=selected_summary_local
                            )
                        )
                    else:
                        st.info("尚未填入 GPT 評分")

                    st.markdown(
                        f"<p style='font-size: 16px;'>已評審題數：{len(evaluated_question_ids)}/{total_count}</p>",
                        unsafe_allow_html=True
                    )
                else:
                    st.info("尚未填入 GPT 評分")
            else:
                st.info("GPT 評審未啟用")

        overview_blocks.append(render_overall)
        overview_blocks.append(render_keyword)

        show_semantic_metric = st.session_state.display_semantic_metric
        if enable_semantic and show_semantic_metric:
            overview_blocks.append(render_semantic)

        overview_blocks.append(render_gpt)

        cols = st.columns(len(overview_blocks))
        for col, renderer in zip(cols, overview_blocks):
            with col:
                renderer()

        # 評估層級配置顯示
        st.markdown("### ⚙️ 評估配置")
        config_col1, config_col2, config_col3 = st.columns(3)

        with config_col1:
            st.metric("關鍵詞匹配", "✅ 啟用", f"權重: {weights['keyword']:.0%}")

        with config_col2:
            status = "✅ 啟用" if enable_semantic else "❌ 停用"
            st.metric("語義相似度", status, f"權重: {weights['semantic']:.0%}")

        with config_col3:
            status = "✅ 啟用" if enable_manual_gpt else "❌ 停用"
            st.metric("GPT 人工評審", status, f"權重: {weights['gpt']:.0%}")

    with tab2:
        st.markdown("### 🤖 GPT 人工評審助手")
        st.info("💡 在這裡生成 prompt → 複製到 ChatGPT → 貼回評分結果 → 所有指標即時更新")

        st.markdown("#### 🧮 GPT 綜合評分維度")
        available_gpt_dims = list(GPT_DIMENSION_LABELS.keys())
        selected_from_widget = st.multiselect(
            "選擇要納入綜合評分的維度",
            options=available_gpt_dims,
            default=st.session_state.gpt_selected_dimensions,
            format_func=lambda x: GPT_DIMENSION_LABELS[x],
            key="gpt_dimension_selector"
        )

        if selected_from_widget:
            st.session_state.gpt_selected_dimensions = selected_from_widget
        else:
            st.warning("至少需要保留一個維度才能計算綜合評分，已沿用前次設定。")
            if not st.session_state.gpt_selected_dimensions:
                st.session_state.gpt_selected_dimensions = available_gpt_dims
            selected_from_widget = st.session_state.gpt_selected_dimensions
        weight_cols = st.columns(len(selected_from_widget)) if selected_from_widget else []
        new_weights = {}
        for col, dim in zip(weight_cols, selected_from_widget):
            with col:
                default_weight = st.session_state.gpt_dimension_weights.get(dim, 0.25)
                new_weights[dim] = st.number_input(
                    GPT_DIMENSION_LABELS[dim],
                    min_value=0.0,
                    value=float(default_weight),
                    step=0.05,
                    key=f"weight_input_{dim}"
                )

        if new_weights:
            st.session_state.gpt_dimension_weights.update(new_weights)

        selected_gpt_dims_tab2 = st.session_state.gpt_selected_dimensions
        selected_gpt_weights_tab2 = get_gpt_dimension_weights(selected_gpt_dims_tab2)
        selected_weight_summary_tab2 = format_gpt_weight_summary(selected_gpt_dims_tab2, selected_gpt_weights_tab2)

        st.caption(
            "系統會自動根據勾選的維度與指定權重重新計算每題與整體的 GPT 綜合評分。"
        )
        st.caption(f"目前加權設定：{selected_weight_summary_tab2}")

        # 評分一致性指導
        with st.expander("📝 評分指標說明", expanded=False):
            st.markdown("""
            #### 🎯 確保評分一致性的使用方法
            
            **1. *嚴謹且固定的評分機制**  
            我們針對四個核心維度（🎯 相關性、📋 完整性、✅ 準確性、🔒 忠誠度）建立了明確且可量化的評估流程，確保每一次評分都在相同標準下執行：
            - (1)固定分級標準：分數區間與比例（如 ≥90% 為滿分）事先定義，模型不得自行調整。
            - (2)明確計算方式：每個維度均以比例公式計算（如貼題句／總句、正確句／(正確＋錯誤) 等），再對照標準分數表。
            - (3)統一標記流程：所有評分皆依「拆句 → 標記 → 計算 → 對照分數段 → 輸出結果」五步驟進行。
            - (4)要求具體數據：模型必須在 reasoning 中列出具體清單與計算結果（如命中要點數、Supported 句等），每一分皆有依據。
            - (5)統一輸出格式：以固定 JSON 欄位呈現評分與 reasoning，禁止新增或刪改結構。
            - (6)人類語氣與驗證提示：在解釋中以自然語氣說明打分原因，並提醒「請列出所有中間計算數據」，強化透明度。
            - (7)一致性提醒：Prompt 開頭註明參考流程，結尾重申「請嚴格依上述步驟與公式執行」，杜絕任意變動。    
            
            **2. 評分指標說明**
            - 🎯 相關性：回答有沒有針對問題來回答
            - 📋 完整性：回答有沒有包含應該要有的重要資訊（與覆蓋率的差異為是否有針對內容做深度補充）
            - ✅ 準確性：回答的內容是不是正確的，有沒有錯誤資訊（與完整性的差異為不管有沒有遺漏，只看說的內容對不對）
            - 🔒 忠誠度：回應內容嚴格源自原始資料，所有陳述皆有資料支撐，未添加任何未檢索後或是其他補充的資訊。 
            
            **3. 評分指標簡單說明**
            - **🎯 相關性**：把回答拆成句子，標記貼題／離題並計算貼題比例。
            - **📋 完整性**：逐項檢查【必須包含的關鍵資訊】，標記命中／部分命中／缺漏。
            - **✅ 準確性**：列出可驗證陳述，標記正確／錯誤／不可驗證，計算正確率。
            - **🔒 忠誠度**：檢查每句是否有資料佐證，標記 Supported／Partially／Unsupported。
            
            **4. 評分指標評分規則**
            - **🎯 相關性**：將每一句話都分成「有回答到題目」或「偏離題意」，算出有幾句是對題目的，然後看比例有多高，高於九成就拿最高分。
            - **📋 完整性**：列出一定要提到的重點，例如「原因」「步驟」「結果」等，檢查哪些完全寫到了（Covered）、哪些只寫到一半（Partially）、哪些沒寫（Missing）。用「全部重點命中的數量＋一半算部分命中」除以重點總數，就能得出一個比例，比例越高分數越高。
            - **✅ 準確性**：把回答裡能查證的每件事都拆出來，看哪些是真的（Correct）、哪些明顯錯了（Incorrect）、哪些查不到（Unverifiable）。計算「正確 ÷ (正確＋錯誤)」，如果正確率達到 95% 就拿滿分。
            - **🔒 忠誠度**：把每一句話都對照原始資料，看有多少句子是「有來源證明」（Supported）、多少句子只是「部分符合」（Partially Supported）、多少是「杜撰或沒依據」（Unsupported）。同樣用「有來源＋一半算部分」除以總句數的方式算比例，比例越高代表越忠實。
            """)
            
            st.success("✅ 遵循以上指導，可確保評分的一致性和準確性！")

        # 選擇要評審的問題
        question_selector = st.selectbox(
            "選擇要評審的問題",
            range(len(results_df)),
            format_func=lambda x: f"問題 {results_df.iloc[x]['序號']}: {results_df.iloc[x]['測試問題'][:40]}..."
        )

        selected_row = results_df.iloc[question_selector]
        # 使用實際序號作為 GPT 評分的 key（而不是 DataFrame index）
        actual_question_id = int(selected_row['序號'])

        # 顯示問題資訊
        st.markdown("#### 📝 問題資訊")
        st.info(f"**問題**: {selected_row['測試問題']}")
        st.success(f"**應回答詞彙**: {selected_row['應回答之詞彙']}")

        # 生�� prompt 區域
        version_col1, version_col2 = st.columns(2)

        with version_col1:
            st.markdown("#### 🔴 原始版本")

            # 顯示原始回答
            with st.expander("查看原始回答"):
                st.text_area("", value=selected_row['ANSWER_ORIGINAL'], height=150, key="orig_answer_view", disabled=True)

            # 生成 Prompt
            prompt_original = generate_gpt_prompt(
                selected_row['測試問題'],
                selected_row['應回答之詞彙'],
                selected_row['ANSWER_ORIGINAL'],
                version="原始",
                question_id=selected_row['序號']
            )

            st.markdown("**📋 GPT Prompt（複製到 ChatGPT）**")
            st.text_area("", value=prompt_original, height=200, key=f"prompt_orig_{question_selector}")

            if st.button("📋 複製 Prompt (原始版本)", key=f"copy_orig_{question_selector}"):
                st.success("✅ Prompt 已複製！請貼到 ChatGPT")

            # 貼上 GPT 回應
            st.markdown("**📥 貼上 ChatGPT 的 JSON 回應**")
            gpt_response_original = st.text_area(
                "",
                height=150,
                key=f"gpt_response_orig_{question_selector}",
                placeholder='貼上 ChatGPT 回應的 JSON，例如：\n{\n  "relevance": 85,\n  "completeness": 90,\n  ...\n}'
            )

            if st.button("✅ 確認並儲存評分 (原始版本)", key=f"save_orig_{question_selector}"):
                if gpt_response_original.strip():
                    parsed = parse_gpt_response(gpt_response_original)
                    if 'error' not in parsed:
                        # 新增：一致性驗證
                        warnings, errors = validate_scoring_consistency(
                            parsed, 
                            selected_row['測試問題'], 
                            selected_row['ANSWER_ORIGINAL']
                        )
                        
                        if errors:
                            st.error("❌ 發現嚴重問題，無法儲存：")
                            for error in errors:
                                st.text(f"• {error}")
                            st.info("💡 請重新請ChatGPT評分，確保包含所有必要欄位")
                        elif warnings:
                            st.warning("⚠️ 發現評分一致性問題：")
                            for warning in warnings:
                                st.text(f"• {warning}")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("🔄 重新評分", key=f"recheck_orig_{question_selector}"):
                                    st.info("建議重新請ChatGPT評分以確保一致性")
                            with col_b:
                                if st.button("📥 仍要儲存", key=f"force_save_orig_{question_selector}"):
                                    st.session_state.gpt_responses_original[actual_question_id] = parsed
                                    st.success("✅ 原始版本評分已儲存！")
                                    
                                    # 自動保存到歷史紀錄
                                    if auto_save_evaluation(
                                        actual_question_id,
                                        results_df,
                                        weights,
                                        selected_gpt_dims_tab2,
                                        selected_gpt_weights_tab2
                                    ):
                                        st.info("💾 已自動保存到歷史紀錄")
                                    
                                    st.rerun()
                        else:
                            st.session_state.gpt_responses_original[actual_question_id] = parsed
                            st.success("✅ 原始版本評分已儲存！評分格式完全正確")
                            
                            # 自動保存到歷史紀錄
                            if auto_save_evaluation(
                                actual_question_id,
                                results_df,
                                weights,
                                selected_gpt_dims_tab2,
                                selected_gpt_weights_tab2
                            ):
                                st.info("💾 已自動保存到歷史紀錄")
                            
                            st.rerun()
                    else:
                        st.error("❌ 無法解析 JSON，請檢查格式")
                        st.text(f"錯誤詳情：{parsed.get('error', '未知錯誤')}")
                else:
                    st.warning("⚠️ 請先貼上 ChatGPT 的回應")

            # 顯示已儲存的評分
            if actual_question_id in st.session_state.gpt_responses_original:
                gpt_data = st.session_state.gpt_responses_original[actual_question_id]
                st.markdown("**📊 已儲存的 GPT 評分**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("相關性", f"{gpt_data.get('relevance', 0)}")
                    st.metric("準確性", f"{gpt_data.get('accuracy', 0)}")
                with col_b:
                    st.metric("完整性", f"{gpt_data.get('completeness', 0)}")
                    st.metric("忠誠度", f"{gpt_data.get('faithfulness', 0)}")
                computed_overall = compute_gpt_overall(
                    gpt_data,
                    selected_gpt_dims_tab2,
                    selected_gpt_weights_tab2
                )
                raw_overall = gpt_data.get('overall')
                try:
                    raw_overall_value = float(raw_overall)
                except (TypeError, ValueError):
                    raw_overall_value = None

                delta_text = None
                if raw_overall_value is not None and abs(computed_overall - raw_overall_value) > 0.01:
                    delta_text = f"{computed_overall - raw_overall_value:+.1f}"

                st.metric("綜合評分", f"{computed_overall:.1f}", delta=delta_text)
                if raw_overall_value is not None and delta_text:
                    st.caption(f"原始 ChatGPT overall：{raw_overall_value:.1f} (未依選擇維度調整)")
                st.caption(f"目前取 {selected_weight_summary_tab2} 的加權平均。")

        with version_col2:
            st.markdown("#### 🟢 優化版本")

            # 顯示優化回答
            with st.expander("查看優化回答"):
                st.text_area("", value=selected_row['ANSWER_OPTIMIZED'], height=150, key="opt_answer_view", disabled=True)

            # 生成 Prompt
            prompt_optimized = generate_gpt_prompt(
                selected_row['測試問題'],
                selected_row['應回答之詞彙'],
                selected_row['ANSWER_OPTIMIZED'],
                version="優化",
                question_id=selected_row['序號']
            )

            st.markdown("**📋 GPT Prompt（複製到 ChatGPT）**")
            st.text_area("", value=prompt_optimized, height=200, key=f"prompt_opt_{question_selector}")

            if st.button("📋 複製 Prompt (優化版本)", key=f"copy_opt_{question_selector}"):
                st.success("✅ Prompt 已複製！請貼到 ChatGPT")

            # 貼上 GPT 回應
            st.markdown("**📥 貼上 ChatGPT 的 JSON 回應**")
            gpt_response_optimized = st.text_area(
                "",
                height=150,
                key=f"gpt_response_opt_{question_selector}",
                placeholder='貼上 ChatGPT 回應的 JSON，例如：\n{\n  "relevance": 85,\n  "completeness": 90,\n  ...\n}'
            )

            if st.button("✅ 確認並儲存評分 (優化版本)", key=f"save_opt_{question_selector}"):
                if gpt_response_optimized.strip():
                    parsed = parse_gpt_response(gpt_response_optimized)
                    if 'error' not in parsed:
                        # 新增：一致性驗證
                        warnings, errors = validate_scoring_consistency(
                            parsed, 
                            selected_row['測試問題'], 
                            selected_row['ANSWER_OPTIMIZED']
                        )
                        
                        if errors:
                            st.error("❌ 發現嚴重問題，無法儲存：")
                            for error in errors:
                                st.text(f"• {error}")
                            st.info("💡 請重新請ChatGPT評分，確保包含所有必要欄位")
                        elif warnings:
                            st.warning("⚠️ 發現評分一致性問題：")
                            for warning in warnings:
                                st.text(f"• {warning}")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                if st.button("🔄 重新評分", key=f"recheck_opt_{question_selector}"):
                                    st.info("建議重新請ChatGPT評分以確保一致性")
                            with col_b:
                                if st.button("📥 仍要儲存", key=f"force_save_opt_{question_selector}"):
                                    st.session_state.gpt_responses_optimized[actual_question_id] = parsed
                                    st.success("✅ 優化版本評分已儲存！")
                                    
                                    # 自動保存到歷史紀錄
                                    if auto_save_evaluation(
                                        actual_question_id,
                                        results_df,
                                        weights,
                                        selected_gpt_dims_tab2,
                                        selected_gpt_weights_tab2
                                    ):
                                        st.info("💾 已自動保存到歷史紀錄")
                                    
                                    st.rerun()
                        else:
                            st.session_state.gpt_responses_optimized[actual_question_id] = parsed
                            st.success("✅ 優化版本評分已儲存！評分格式完全正確")
                            
                            # 自動保存到歷史紀錄
                            if auto_save_evaluation(
                                actual_question_id,
                                results_df,
                                weights,
                                selected_gpt_dims_tab2,
                                selected_gpt_weights_tab2
                            ):
                                st.info("💾 已自動保存到歷史紀錄")
                            
                            st.rerun()
                    else:
                        st.error("❌ 無法解析 JSON，請檢查格式")
                        st.text(f"錯誤詳情：{parsed.get('error', '未知錯誤')}")
                else:
                    st.warning("⚠️ 請先貼上 ChatGPT 的回應")

            # 顯示已儲存的評分
            if actual_question_id in st.session_state.gpt_responses_optimized:
                gpt_data = st.session_state.gpt_responses_optimized[actual_question_id]
                st.markdown("**📊 已儲存的 GPT 評分**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("相關性", f"{gpt_data.get('relevance', 0)}")
                    st.metric("準確性", f"{gpt_data.get('accuracy', 0)}")
                with col_b:
                    st.metric("完整性", f"{gpt_data.get('completeness', 0)}")
                    st.metric("忠誠度", f"{gpt_data.get('faithfulness', 0)}")
                computed_overall = compute_gpt_overall(
                    gpt_data,
                    selected_gpt_dims_tab2,
                    selected_gpt_weights_tab2
                )
                raw_overall = gpt_data.get('overall')
                try:
                    raw_overall_value = float(raw_overall)
                except (TypeError, ValueError):
                    raw_overall_value = None

                delta_text = None
                if raw_overall_value is not None and abs(computed_overall - raw_overall_value) > 0.01:
                    delta_text = f"{computed_overall - raw_overall_value:+.1f}"

                st.metric("綜合評分", f"{computed_overall:.1f}", delta=delta_text)
                if raw_overall_value is not None and delta_text:
                    st.caption(f"原始 ChatGPT overall：{raw_overall_value:.1f} (未依選擇維度調整)")
                st.caption(f"目前取 {selected_weight_summary_tab2} 的加權平均。")

        # 批次操作提示
        st.markdown("---")
        st.markdown("### 📊 批次評審進度")

        total_questions = len(results_df)
        evaluated_original = len(st.session_state.gpt_responses_original)
        evaluated_optimized = len(st.session_state.gpt_responses_optimized)

        progress_col1, progress_col2 = st.columns(2)

        with progress_col1:
            st.metric("原始版本已評審", f"{evaluated_original}/{total_questions}",
                     f"{evaluated_original/total_questions*100:.1f}%")
            st.progress(evaluated_original / total_questions)

        with progress_col2:
            st.metric("優化版本已評審", f"{evaluated_optimized}/{total_questions}",
                     f"{evaluated_optimized/total_questions*100:.1f}%")
            st.progress(evaluated_optimized / total_questions)

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("💾 手動保存全部到歷史紀錄", key="manual_save_all", type="primary", use_container_width=True):
                saved_count = 0
                for idx in range(len(results_df)):
                    if auto_save_evaluation(
                        idx,
                        results_df,
                        weights,
                        selected_gpt_dims_tab2,
                        selected_gpt_weights_tab2
                    ):
                        saved_count += 1
                st.success(f"✅ 成功保存 {saved_count} 筆評估到歷史紀錄")
                st.rerun()

        with col_btn2:
            if st.button("🔄 清除所有 GPT 評分", key="clear_all_gpt", use_container_width=True):
                st.session_state.gpt_responses_original = {}
                st.session_state.gpt_responses_optimized = {}
                st.success("✅ 已清除所有 GPT 評分")
                st.rerun()

    with tab3:
        st.markdown("### 📈 詳細對比分析")
        st.info("整合三層評估結果的完整對比（包含您提供的 GPT 評分）")

        # 建立多層級對比表格
        comparison_data = []

        metrics = [
            ('關鍵詞覆蓋率', 'KEYWORD_COVERAGE'),
        ]

        if enable_semantic:
            metrics.append(('語義相似度', 'SEMANTIC_SIMILARITY'))

        if enable_manual_gpt:
            metrics.append(('GPT 評分', 'GPT_OVERALL'))

        metrics.append(('綜合評分', 'FINAL_SCORE'))

        for metric_name, metric_key in metrics:
            comparison_data.append({
                '評估指標': f'🔴 原始版本 - {metric_name}',
                '平均分數': f"{results_df[f'{metric_key}_ORIGINAL'].mean():.1f}",
                '最高分': f"{results_df[f'{metric_key}_ORIGINAL'].max():.1f}",
                '最低分': f"{results_df[f'{metric_key}_ORIGINAL'].min():.1f}",
                '標準差': f"{results_df[f'{metric_key}_ORIGINAL'].std():.1f}"
            })

            comparison_data.append({
                '評估指標': f'🟢 優化版本 - {metric_name}',
                '平均分數': f"{results_df[f'{metric_key}_OPTIMIZED'].mean():.1f}",
                '最高分': f"{results_df[f'{metric_key}_OPTIMIZED'].max():.1f}",
                '最低分': f"{results_df[f'{metric_key}_OPTIMIZED'].min():.1f}",
                '標準差': f"{results_df[f'{metric_key}_OPTIMIZED'].std():.1f}"
            })

            improvement = results_df[f'{metric_key}_OPTIMIZED'].mean() - results_df[f'{metric_key}_ORIGINAL'].mean()
            comparison_data.append({
                '評估指標': f'📊 改善幅度 - {metric_name}',
                '平均分數': f"{improvement:+.1f}",
                '最高分': "-",
                '最低分': "-",
                '標準差': "-"
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # 雷達圖對比
        st.markdown("### 🎯 多維度雷達圖對比")

        categories = ['關鍵詞覆蓋率']
        original_scores = [results_df['KEYWORD_COVERAGE_ORIGINAL'].mean()]
        optimized_scores = [results_df['KEYWORD_COVERAGE_OPTIMIZED'].mean()]

        if enable_semantic:
            categories.append('語義相似度')
            original_scores.append(results_df['SEMANTIC_SIMILARITY_ORIGINAL'].mean())
            optimized_scores.append(results_df['SEMANTIC_SIMILARITY_OPTIMIZED'].mean())

        if enable_manual_gpt:
            categories.append('GPT 評分')
            original_scores.append(results_df['GPT_OVERALL_ORIGINAL'].mean())
            optimized_scores.append(results_df['GPT_OVERALL_OPTIMIZED'].mean())

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=original_scores + [original_scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='原始版本',
            line_color='#e57373'
        ))

        fig_radar.add_trace(go.Scatterpolar(
            r=optimized_scores + [optimized_scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='優化版本',
            line_color='#81c784'
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    with tab4:
        st.markdown("### 🔤 語義差異分析")

        if not enable_semantic:
            st.warning("語義相似度功能未啟用，請在左側勾選並確認環境已安裝 sentence-transformers。")
        elif not st.session_state.evaluator_instance or not st.session_state.evaluator_instance.enable_semantic:
            st.warning("語義模型尚未載入成功，請重新執行評估或檢查環境設定。")
        else:
            evaluator = st.session_state.evaluator_instance

            semantic_selector = st.selectbox(
                "選擇要分析的問題",
                range(len(results_df)),
                format_func=lambda x: f"問題 {results_df.iloc[x]['序號']}: {results_df.iloc[x]['測試問題'][:40]}...",
                key="semantic_selector"
            )

            row = results_df.iloc[semantic_selector]
            question_id = int(row['序號'])
            reference_text = row['應回答之詞彙']
            answer_original = row['ANSWER_ORIGINAL']
            answer_optimized = row['ANSWER_OPTIMIZED']

            st.markdown(f"#### 🧾 問題 {question_id}: {row['測試問題']}")
            with st.expander("應回答之詞彙 / 參考內容", expanded=False):
                st.write(reference_text)

            with st.expander("查看原始版本回答", expanded=False):
                st.write(answer_original)

            with st.expander("查看優化版本回答", expanded=False):
                st.write(answer_optimized)

            st.info(
                "語義相似度是將『應回答之詞彙』與實際回答做向量比對，因此即使關鍵詞皆命中，"
                "若句型、用詞或補充內容與參考資料差異大，分數仍會降低。"
            )

            ideal_lines = format_reference_to_list(reference_text)
            if ideal_lines:
                st.markdown("**理想語義示例（與參考內容對齊的寫法）**")
                ideal_text = "\n".join([f"{idx + 1}. {line}" for idx, line in enumerate(ideal_lines)])
                st.code(ideal_text, language="text")

            score_col1, score_col2 = st.columns(2)

            keywords = evaluator.extract_keywords(reference_text)
            orig_kw_score, orig_matched, orig_details = evaluator.calculate_keyword_coverage(answer_original, keywords)
            opt_kw_score, opt_matched, opt_details = evaluator.calculate_keyword_coverage(answer_optimized, keywords)

            orig_sem_score, orig_sem_details = evaluator.calculate_semantic_similarity(reference_text, answer_original)
            opt_sem_score, opt_sem_details = evaluator.calculate_semantic_similarity(reference_text, answer_optimized)

            ref_sentences = split_into_sentences(reference_text)
            orig_sentence_scores = compute_sentence_similarity(evaluator, ref_sentences, answer_original)
            opt_sentence_scores = compute_sentence_similarity(evaluator, ref_sentences, answer_optimized)

            def build_sentence_table(data):
                if not data:
                    return pd.DataFrame(columns=["句子", "與回答相似度"])
                sorted_data = sorted(data, key=lambda x: x[1])
                top_items = sorted_data[:3]
                return pd.DataFrame([
                    {"句子": sent, "與回答相似度": f"{score:.1f}%"} for sent, score in top_items
                ])

            with score_col1:
                st.markdown("##### 🔴 原始版本")
                st.metric("語義相似度", f"{row['SEMANTIC_SIMILARITY_ORIGINAL']:.1f}%")
                st.caption(
                    f"餘弦相似度：{orig_sem_details.get('raw_similarity', 0):.3f}｜參考長度：{orig_sem_details.get('reference_length', len(reference_text))}｜回答長度：{orig_sem_details.get('answer_length', len(str(answer_original)))}"
                )

                st.markdown("**缺漏關鍵詞**")
                missing_keywords = orig_details.get('missing_list', [])
                if missing_keywords:
                    st.write('、'.join(missing_keywords))
                else:
                    st.success("關鍵詞皆已覆蓋")

                st.markdown("**低相似度參考句**")
                sentence_table = build_sentence_table(orig_sentence_scores)
                if sentence_table.empty:
                    st.info("參考資料無可比較句子或回答為空。")
                else:
                    st.table(sentence_table)

                if missing_keywords:
                    st.markdown(
                        "**建議**：補充上述缺漏關鍵詞，或將回答中的敘述調整成貼近參考資料的語句，以提升語義覆蓋率。"
                    )

            with score_col2:
                st.markdown("##### 🟢 優化版本")
                st.metric("語義相似度", f"{row['SEMANTIC_SIMILARITY_OPTIMIZED']:.1f}%")
                st.caption(
                    f"餘弦相似度：{opt_sem_details.get('raw_similarity', 0):.3f}｜參考長度：{opt_sem_details.get('reference_length', len(reference_text))}｜回答長度：{opt_sem_details.get('answer_length', len(str(answer_optimized)))}"
                )

                st.markdown("**缺漏關鍵詞**")
                missing_keywords_opt = opt_details.get('missing_list', [])
                if missing_keywords_opt:
                    st.write('、'.join(missing_keywords_opt))
                else:
                    st.success("關鍵詞皆已覆蓋")

                st.markdown("**低相似度參考句**")
                sentence_table_opt = build_sentence_table(opt_sentence_scores)
                if sentence_table_opt.empty:
                    st.info("參考資料無可比較句子或回答為空。")
                else:
                    st.table(sentence_table_opt)

                if missing_keywords_opt:
                    st.markdown(
                        "**建議**：補足缺漏詞彙，並比對低相似句子的資訊重點，讓回答更貼近參考內容。"
                    )

            st.markdown("---")
            st.markdown("#### 🧭 參考如何撰寫更理想的回答？")
            improvement_points = []

            if missing_keywords:
                improvement_points.append("原始版本漏掉的關鍵詞：" + '、'.join(missing_keywords))
            if missing_keywords_opt:
                improvement_points.append("優化版本仍需補充的關鍵詞：" + '、'.join(missing_keywords_opt))

            if orig_sentence_scores:
                lowest_orig = sorted(orig_sentence_scores, key=lambda x: x[1])[:1]
                for sent, score in lowest_orig:
                    improvement_points.append(f"原始版與參考句「{sent}」僅 {score:.1f}% 相似，可加入相對應細節。")

            if opt_sentence_scores:
                lowest_opt = sorted(opt_sentence_scores, key=lambda x: x[1])[:1]
                for sent, score in lowest_opt:
                    improvement_points.append(f"優化版與參考句「{sent}」僅 {score:.1f}% 相似，可再貼近原始描述。")

            if improvement_points:
                for item in improvement_points:
                    st.markdown(f"- {item}")
            else:
                st.success("兩個版本與參考內容高度一致，無明顯缺漏。")

    with tab5:
        st.markdown("### 💬 GPT評分導覽")
        st.info("瀏覽所有測試問題的詳細評估結果")

        # 篩選選項
        filter_option = st.selectbox(
            "篩選顯示",
            ["所有問題", "顯著改善", "略有改善", "無變化", "效果退步", "已有 GPT 評分", "未有 GPT 評分"]
        )

        evaluated_question_ids = set(st.session_state.gpt_responses_original.keys()) | set(
            st.session_state.gpt_responses_optimized.keys()
        )

        # 根據條件篩選
        if filter_option == "顯著改善":
            filtered_df = results_df[results_df['FINAL_IMPROVEMENT'] >= improvement_threshold]
        elif filter_option == "略有改善":
            filtered_df = results_df[(results_df['FINAL_IMPROVEMENT'] > 0) & (results_df['FINAL_IMPROVEMENT'] < improvement_threshold)]
        elif filter_option == "無變化":
            filtered_df = results_df[results_df['FINAL_IMPROVEMENT'] == 0]
        elif filter_option == "效果退步":
            filtered_df = results_df[results_df['FINAL_IMPROVEMENT'] < 0]
        elif filter_option == "已有 GPT 評分":
            filtered_df = results_df[results_df['序號'].astype(int).isin(evaluated_question_ids)]
        elif filter_option == "未有 GPT 評分":
            filtered_df = results_df[~results_df['序號'].astype(int).isin(evaluated_question_ids)]
        else:
            filtered_df = results_df

        st.info(f"顯示 {len(filtered_df)} / {len(results_df)} 個問題")

        # 顯示問題列表
        for idx, row in filtered_df.iterrows():
            question_id = int(row['序號'])
            improvement = row['FINAL_IMPROVEMENT']
            improvement_icon = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            
            with st.expander(f"{improvement_icon} 問題 {row['序號']}: {row['測試問題'][:50]}... (改善:{improvement:+.1f})"):
                st.markdown(f"**測試問題**: {row['測試問題']}")
                st.markdown(f"**應回答詞彙**: {row['應回答之詞彙']}")

                # 基礎評分對比
                score_col1, score_col2, score_col3, score_col4 = st.columns(4)

                with score_col1:
                    st.metric(
                        "關鍵詞覆蓋率",
                        f"{row['KEYWORD_COVERAGE_OPTIMIZED']:.1f}%",
                        f"{row['KEYWORD_COVERAGE_OPTIMIZED'] - row['KEYWORD_COVERAGE_ORIGINAL']:.1f}%"
                    )

                with score_col2:
                    if enable_semantic:
                        st.metric(
                            "語義相似度",
                            f"{row['SEMANTIC_SIMILARITY_OPTIMIZED']:.1f}%",
                            f"{row['SEMANTIC_SIMILARITY_OPTIMIZED'] - row['SEMANTIC_SIMILARITY_ORIGINAL']:.1f}%"
                        )

                with score_col3:
                    if enable_manual_gpt and question_id in st.session_state.gpt_responses_optimized:
                        st.metric(
                            "GPT 評分",
                            f"{row['GPT_OVERALL_OPTIMIZED']:.1f}",
                            f"{row['GPT_OVERALL_OPTIMIZED'] - row['GPT_OVERALL_ORIGINAL']:.1f}"
                        )
                    elif enable_manual_gpt and question_id in st.session_state.gpt_responses_original:
                        st.info("僅原始版已評審")
                    elif enable_manual_gpt:
                        st.warning("未評審")

                with score_col4:
                    st.metric(
                        "📊 綜合評分",
                        f"{row['FINAL_SCORE_OPTIMIZED']:.1f}",
                        f"{row['FINAL_IMPROVEMENT']:.1f}"
                    )
                
                # 各指標變化原因分析
                st.markdown("---")
                st.markdown("#### 🔍 各指標變化原因分析")
                
                # 關鍵詞覆蓋率分析
                keyword_change = row['KEYWORD_COVERAGE_OPTIMIZED'] - row['KEYWORD_COVERAGE_ORIGINAL']
                if abs(keyword_change) > 0.1:  # 只顯示有變化的指標
                    st.markdown("**🎯 關鍵詞覆蓋率變化分析**")
                    
                    kw_col1, kw_col2 = st.columns(2)
                    with kw_col1:
                        st.info(f"🔴 **原始版本**: {row['KEYWORD_COVERAGE_ORIGINAL']:.1f}%")
                    with kw_col2:
                        if keyword_change > 0:
                            st.success(f"🟢 **優化版本**: {row['KEYWORD_COVERAGE_OPTIMIZED']:.1f}% (提升 {keyword_change:.1f}%)")
                        else:
                            st.error(f"🔴 **優化版本**: {row['KEYWORD_COVERAGE_OPTIMIZED']:.1f}% (下降 {abs(keyword_change):.1f}%)")
                    
                    # 關鍵詞變化原因分析
                    if st.session_state.evaluator_instance:
                        evaluator = st.session_state.evaluator_instance
                        keywords = evaluator.extract_keywords(row['應回答之詞彙'])
                        
                        # 原始版本關鍵詞分析
                        orig_score, orig_matched, orig_details = evaluator.calculate_keyword_coverage(
                            row['ANSWER_ORIGINAL'], keywords
                        )
                        
                        # 優化版本關鍵詞分析  
                        opt_score, opt_matched, opt_details = evaluator.calculate_keyword_coverage(
                            row['ANSWER_OPTIMIZED'], keywords
                        )
                        
                        # 詳細關鍵詞覆蓋分析
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.write("**原始版本 - 關鍵詞詳情:**")
                            orig_found = orig_matched or orig_details.get('found_list', [])
                            if orig_found:
                                st.write("✅ **已覆蓋關鍵詞:**")
                                for kw in orig_found:
                                    st.write(f"  • {kw}")

                            orig_missing = [kw for kw in keywords if kw not in orig_found]
                            if orig_missing:
                                st.write("❌ **未覆蓋關鍵詞:**")
                                for kw in orig_missing:
                                    st.write(f"  • {kw}")

                        with detail_col2:
                            st.write("**優化版本 - 關鍵詞詳情:**")
                            opt_found = opt_matched or opt_details.get('found_list', [])
                            if opt_found:
                                st.write("✅ **已覆蓋關鍵詞:**")
                                for kw in opt_found:
                                    st.write(f"  • {kw}")
                            
                            opt_missing = [kw for kw in keywords if kw not in opt_found]
                            if opt_missing:
                                st.write("❌ **未覆蓋關鍵詞:**")
                                for kw in opt_missing:
                                    st.write(f"  • {kw}")
                        
                        # 變化摘要
                        if keyword_change > 0:
                            # 提升原因
                            newly_found = set(opt_found) - set(orig_found)
                            if newly_found:
                                st.success(f"🆕 **新增命中關鍵詞**: {', '.join(newly_found)}")
                            else:
                                st.success("✅ **改善原因**: 優化版本更完整地包含了既有關鍵詞")
                        elif keyword_change < 0:
                            # 下降原因
                            lost_keywords = set(orig_found) - set(opt_found)
                            if lost_keywords:
                                st.error(f"📉 **遺失關鍵詞**: {', '.join(lost_keywords)}")
                            else:
                                st.error("❌ **下降原因**: 優化版本可能過度簡化了關鍵資訊")
                        else:
                            st.info("➡️ **關鍵詞覆蓋率無變化**")
                
                # 語義相似度分析
                if enable_semantic:
                    semantic_change = row['SEMANTIC_SIMILARITY_OPTIMIZED'] - row['SEMANTIC_SIMILARITY_ORIGINAL']
                    if abs(semantic_change) > 0.1:  # 只顯示有變化的指標
                        st.markdown("**🔤 語義相似度變化分析**")
                        
                        sem_col1, sem_col2 = st.columns(2)
                        with sem_col1:
                            st.info(f"🔴 **原始版本**: {row['SEMANTIC_SIMILARITY_ORIGINAL']:.1f}%")
                        with sem_col2:
                            if semantic_change > 0:
                                st.success(f"🟢 **優化版本**: {row['SEMANTIC_SIMILARITY_OPTIMIZED']:.1f}% (提升 {semantic_change:.1f}%)")
                            else:
                                st.error(f"🟢 **優化版本**: {row['SEMANTIC_SIMILARITY_OPTIMIZED']:.1f}% (下降 {abs(semantic_change):.1f}%)")
                        
                        # 語義相似度變化原因分析
                        if st.session_state.evaluator_instance and st.session_state.evaluator_instance.enable_semantic:
                            evaluator = st.session_state.evaluator_instance
                            
                            # 計算詳細語義相似度
                            orig_sem_score, orig_sem_details = evaluator.calculate_semantic_similarity(
                                row['應回答之詞彙'], row['ANSWER_ORIGINAL']
                            )
                            opt_sem_score, opt_sem_details = evaluator.calculate_semantic_similarity(
                                row['應回答之詞彙'], row['ANSWER_OPTIMIZED']
                            )
                            
                            if semantic_change > 0:
                                st.success(
                                    "✅ **語義改善原因**: 優化版本更貼近參考內容的表達方式和詞彙選擇"
                                )
                                st.caption(f"原始餘弦相似度: {orig_sem_details.get('raw_similarity', 0):.3f} → 優化後: {opt_sem_details.get('raw_similarity', 0):.3f}")
                            else:
                                st.error(
                                    "❌ **語義下降原因**: 優化版本可能使用了較不常見的詞彙或表達方式"
                                )
                                st.caption(f"原始餘弦相似度: {orig_sem_details.get('raw_similarity', 0):.3f} → 優化後: {opt_sem_details.get('raw_similarity', 0):.3f}")

                # 詳細GPT評分分析
                if enable_manual_gpt and (question_id in st.session_state.gpt_responses_original or question_id in st.session_state.gpt_responses_optimized):
                    st.markdown("---")
                    st.markdown("#### 🤖 GPT 詳細評分分析")
                    
                    # GPT四維度對比
                    gpt_orig = st.session_state.gpt_responses_original.get(question_id, {})
                    gpt_opt = st.session_state.gpt_responses_optimized.get(question_id, {})
                    
                    gpt_col1, gpt_col2, gpt_col3, gpt_col4 = st.columns(4)
                    
                    with gpt_col1:
                        orig_rel = gpt_orig.get('relevance', 0)
                        opt_rel = gpt_opt.get('relevance', 0)
                        if orig_rel > 0 and opt_rel > 0:
                            st.metric("🎯 相關性", f"{opt_rel}", f"{opt_rel - orig_rel:+.0f}")
                        elif opt_rel > 0:
                            st.metric("🎯 相關性", f"{opt_rel}", "優化版")
                        elif orig_rel > 0:
                            st.metric("🎯 相關性", f"{orig_rel}", "原始版")
                    
                    with gpt_col2:
                        orig_comp = gpt_orig.get('completeness', 0)
                        opt_comp = gpt_opt.get('completeness', 0)
                        if orig_comp > 0 and opt_comp > 0:
                            st.metric("📋 完整性", f"{opt_comp}", f"{opt_comp - orig_comp:+.0f}")
                        elif opt_comp > 0:
                            st.metric("📋 完整性", f"{opt_comp}", "優化版")
                        elif orig_comp > 0:
                            st.metric("📋 完整性", f"{orig_comp}", "原始版")
                    
                    with gpt_col3:
                        orig_acc = gpt_orig.get('accuracy', 0)
                        opt_acc = gpt_opt.get('accuracy', 0)
                        if orig_acc > 0 and opt_acc > 0:
                            st.metric("✅ 準確性", f"{opt_acc}", f"{opt_acc - orig_acc:+.0f}")
                        elif opt_acc > 0:
                            st.metric("✅ 準確性", f"{opt_acc}", "優化版")
                        elif orig_acc > 0:
                            st.metric("✅ 準確性", f"{orig_acc}", "原始版")
                    
                    with gpt_col4:
                        orig_faith = gpt_orig.get('faithfulness', 0)
                        opt_faith = gpt_opt.get('faithfulness', 0)
                        if orig_faith > 0 and opt_faith > 0:
                            st.metric("🔒 忠誠度", f"{opt_faith}", f"{opt_faith - orig_faith:+.0f}")
                        elif opt_faith > 0:
                            st.metric("🔒 忠誠度", f"{opt_faith}", "優化版")
                        elif orig_faith > 0:
                            st.metric("🔒 忠誠度", f"{orig_faith}", "原始版")
                    
                    # GPT評分原因分析
                    if gpt_orig or gpt_opt:
                        st.markdown("#### 💭 評分原因詳細分析")
                        
                        reasoning_col1, reasoning_col2 = st.columns(2)
                        
                        with reasoning_col1:
                            if gpt_orig:
                                st.markdown("##### 🔴 原始版本評分原因")
                                
                                if gpt_orig.get('relevance_reasoning'):
                                    st.markdown(f"**🎯 相關性 ({gpt_orig.get('relevance', 0)}分)**")
                                    st.info(gpt_orig.get('relevance_reasoning', '無詳細說明'))
                                
                                if gpt_orig.get('completeness_reasoning'):
                                    st.markdown(f"**📋 完整性 ({gpt_orig.get('completeness', 0)}分)**")
                                    st.info(gpt_orig.get('completeness_reasoning', '無詳細說明'))
                                
                                if gpt_orig.get('accuracy_reasoning'):
                                    st.markdown(f"**✅ 準確性 ({gpt_orig.get('accuracy', 0)}分)**")
                                    st.info(gpt_orig.get('accuracy_reasoning', '無詳細說明'))
                                
                                if gpt_orig.get('faithfulness_reasoning'):
                                    st.markdown(f"**🔒 忠誠度 ({gpt_orig.get('faithfulness', 0)}分)**")
                                    st.info(gpt_orig.get('faithfulness_reasoning', '無詳細說明'))
                            else:
                                st.info("原始版本尚未進行GPT評分")
                        
                        with reasoning_col2:
                            if gpt_opt:
                                st.markdown("##### 🟢 優化版本評分原因")
                                
                                if gpt_opt.get('relevance_reasoning'):
                                    st.markdown(f"**🎯 相關性 ({gpt_opt.get('relevance', 0)}分)**")
                                    st.success(gpt_opt.get('relevance_reasoning', '無詳細說明'))
                                
                                if gpt_opt.get('completeness_reasoning'):
                                    st.markdown(f"**📋 完整性 ({gpt_opt.get('completeness', 0)}分)**")
                                    st.success(gpt_opt.get('completeness_reasoning', '無詳細說明'))
                                
                                if gpt_opt.get('accuracy_reasoning'):
                                    st.markdown(f"**✅ 準確性 ({gpt_opt.get('accuracy', 0)}分)**")
                                    st.success(gpt_opt.get('accuracy_reasoning', '無詳細說明'))
                                
                                if gpt_opt.get('faithfulness_reasoning'):
                                    st.markdown(f"**🔒 忠誠度 ({gpt_opt.get('faithfulness', 0)}分)**")
                                    st.success(gpt_opt.get('faithfulness_reasoning', '無詳細說明'))
                            else:
                                st.info("優化版本尚未進行GPT評分")

                    # 改進建議
                    if gpt_orig and gpt_opt:
                        st.markdown("---")
                        st.markdown("#### 📈 改進分析與建議")
                        
                        improvements = []
                        concerns = []
                        
                        # 分析各維度變化
                        dimensions = [
                            ('相關性', 'relevance', '🎯'),
                            ('完整性', 'completeness', '📋'), 
                            ('準確性', 'accuracy', '✅'),
                            ('忠誠度', 'faithfulness', '🔒')
                        ]

                        for dim_name, dim_key, dim_icon in dimensions:
                            orig_score = gpt_orig.get(dim_key, 0)
                            opt_score = gpt_opt.get(dim_key, 0)
                            diff = opt_score - orig_score

                            orig_reason = (gpt_orig.get(f"{dim_key}_reasoning") or "").strip()
                            opt_reason = (gpt_opt.get(f"{dim_key}_reasoning") or "").strip()
                            reason_text = opt_reason if diff >= 0 else orig_reason
                            if reason_text:
                                reason_text = reason_text.replace("\n", "  \n")
                            else:
                                reason_text = "（GPT 未提供詳細說明）"

                            if diff > 5:
                                improvements.append(
                                    f"{dim_icon} **{dim_name}顯著提升** (+{diff:.0f}分)  \n"
                                    f"> {reason_text}"
                                )
                            elif diff < -5:
                                concerns.append(
                                    f"{dim_icon} **{dim_name}有所下降** ({diff:+.0f}分)  \n"
                                    f"> {reason_text}"
                                )
                        
                        if improvements:
                            st.markdown("**✅ 主要改進**")
                            for improvement in improvements:
                                st.markdown(f"- {improvement}")
                        
                        if concerns:
                            st.markdown("**⚠️ 需要注意**")
                            for concern in concerns:
                                st.markdown(f"- {concern}")
                        
                        if not improvements and not concerns:
                            st.info("💡 兩個版本在各維度表現相當，差異不大")

                    # 原始 JSON 參考
                    json_col1, json_col2 = st.columns(2)
                    with json_col1:
                        if gpt_orig:
                            st.markdown("**🗒️ 原始版本 GPT JSON**")
                            st.code(json.dumps(gpt_orig, ensure_ascii=False, indent=2), language="json")
                        else:
                            st.info("原始版本尚未貼上 GPT JSON")
                    with json_col2:
                        if gpt_opt:
                            st.markdown("**🗒️ 優化版本 GPT JSON**")
                            st.code(json.dumps(gpt_opt, ensure_ascii=False, indent=2), language="json")
                        else:
                            st.info("優化版本尚未貼上 GPT JSON")
                    
                    # 顯示原始回答對比
                    st.markdown("---")
                    st.markdown("#### 📝 回答內容對比")
                    
                    answer_col1, answer_col2 = st.columns(2)
                    
                    with answer_col1:
                        st.markdown("##### 🔴 原始版本回答")
                        # 使用可收合的容器來取代 expander
                        show_orig = st.checkbox("顯示原始回答", key=f"show_orig_{question_id}")
                        if show_orig:
                            st.text_area("", value=row['ANSWER_ORIGINAL'], height=150, key=f"orig_answer_{question_id}", disabled=True)
                    
                    with answer_col2:
                        st.markdown("##### 🟢 優化版本回答")
                        # 使用可收合的容器來取代 expander
                        show_opt = st.checkbox("顯示優化回答", key=f"show_opt_{question_id}")
                        if show_opt:
                            st.text_area("", value=row['ANSWER_OPTIMIZED'], height=150, key=f"opt_answer_{question_id}", disabled=True)

    with tab6:
        st.markdown("### 🎯 關鍵詞分析")
        st.info("🔍 逐題檢視關鍵詞覆蓋率的詳細表現，包含已覆蓋和未覆蓋的關鍵詞列表")
        
        if 'comparison_results' in st.session_state and st.session_state.evaluator_instance:
            results_df = st.session_state.comparison_results
            evaluator = st.session_state.evaluator_instance
            
            # 題目選擇器
            keyword_selector = st.selectbox(
                "選擇要分析的問題",
                range(len(results_df)),
                format_func=lambda x: f"問題 {results_df.iloc[x]['序號']}: {results_df.iloc[x]['測試問題'][:40]}...",
                key="keyword_selector"
            )
            
            row = results_df.iloc[keyword_selector]
            question_id = int(row['序號'])
            reference_text = row['應回答之詞彙']
            answer_original = row['ANSWER_ORIGINAL']
            answer_optimized = row['ANSWER_OPTIMIZED']
            
            st.markdown(f"#### 📝 問題 {question_id}: {row['測試問題']}")
            
            with st.expander("應回答之詞彙 / 參考內容", expanded=False):
                st.write(reference_text)
            
            with st.expander("查看原始版本回答", expanded=False):
                st.write(answer_original)
            
            with st.expander("查看優化版本回答", expanded=False):
                st.write(answer_optimized)
            
            st.info(
                "📈 關鍵詞覆蓋率是檢查回答中是否包含『應回答之詞彙』中的關鍵詞彙。"
                "本分析將清楚顯示哪些詞彙已覆蓋、哪些尚未覆蓋。"
            )
            
            # 提取關鍵詞和計算覆蓋率
            keywords = evaluator.extract_keywords(reference_text)
            orig_kw_score, orig_matched, orig_details = evaluator.calculate_keyword_coverage(answer_original, keywords)
            opt_kw_score, opt_matched, opt_details = evaluator.calculate_keyword_coverage(answer_optimized, keywords)
            
            # 顯示所有關鍵詞列表
            if keywords:
                st.markdown("**🗒️ 所有關鍵詞列表**")
                keyword_list = "、".join(keywords)
                st.code(keyword_list, language="text")
            else:
                st.warning("⚠️ 無法提取關鍵詞")
                st.stop()
            
            # 對比分析與覆蓋率說明
            total_keywords = len(keywords)
            orig_found = orig_matched or []
            opt_found = opt_matched or []
            orig_missing = (orig_details or {}).get('missing_list', [])
            opt_missing = (opt_details or {}).get('missing_list', [])
            found_count = len(orig_found)
            opt_found_count = len(opt_found)
            orig_hit_pct = (found_count / total_keywords * 100) if total_keywords else 0.0
            opt_hit_pct = (opt_found_count / total_keywords * 100) if total_keywords else 0.0
            change = row['KEYWORD_COVERAGE_OPTIMIZED'] - row['KEYWORD_COVERAGE_ORIGINAL']

            score_col1, score_col2 = st.columns(2)

            with score_col1:
                st.markdown("##### 🔴 原始版本")
                st.metric("關鍵詞覆蓋率", f"{row['KEYWORD_COVERAGE_ORIGINAL']:.1f}%")

                st.caption(f"命中關鍵詞數量：{found_count}/{total_keywords}（命中率 {orig_hit_pct:.1f}%）")

                st.markdown("**📊 覆蓋率計算說明**")
                if total_keywords:
                    st.write(
                        f"應覆蓋關鍵詞共 {total_keywords} 個，實際命中 {found_count} 個，"
                        f"覆蓋率 = {found_count}/{total_keywords} × 100 = {row['KEYWORD_COVERAGE_ORIGINAL']:.1f}%。"
                    )
                else:
                    st.warning("本題未能擷取到可用的關鍵詞，因此無法計算覆蓋率。")

                st.markdown("**✅ 已覆蓋關鍵詞**")
                if orig_found:
                    for kw in orig_found:
                        st.success(f"• {kw}")
                else:
                    st.info("無命中關鍵詞")

                st.markdown("**❌ 未覆蓋關鍵詞**")
                if orig_missing:
                    for kw in orig_missing:
                        st.error(f"• {kw}")
                    st.markdown(
                        "**建議**：在回答中加入上述缺漏的關鍵詞，提高覆蓋率。"
                    )
                else:
                    st.success("全部關鍵詞已覆蓋！")

            with score_col2:
                st.markdown("##### 🟢 優化版本")
                st.metric("關鍵詞覆蓋率", f"{row['KEYWORD_COVERAGE_OPTIMIZED']:.1f}%", f"{change:+.1f}%")

                st.caption(f"命中關鍵詞數量：{opt_found_count}/{total_keywords}（命中率 {opt_hit_pct:.1f}%）")

                st.markdown("**📊 覆蓋率計算說明**")
                if total_keywords:
                    st.write(
                        f"應覆蓋關鍵詞共 {total_keywords} 個，優化版本命中 {opt_found_count} 個，"
                        f"覆蓋率 = {opt_found_count}/{total_keywords} × 100 = {row['KEYWORD_COVERAGE_OPTIMIZED']:.1f}%。"
                    )
                else:
                    st.warning("本題未能擷取到可用的關鍵詞，因此無法計算覆蓋率。")

                st.markdown("**✅ 已覆蓋關鍵詞**")
                if opt_found:
                    for kw in opt_found:
                        st.success(f"• {kw}")
                else:
                    st.info("無命中關鍵詞")

                st.markdown("**❌ 未覆蓋關鍵詞**")
                if opt_missing:
                    for kw in opt_missing:
                        st.error(f"• {kw}")
                    st.markdown(
                        "**建議**：這些詞彙仍需要被提及以進一步提升覆蓋率。"
                    )
                else:
                    st.success("全部關鍵詞已覆蓋！")

            st.markdown("---")

            st.markdown("#### 📋 關鍵詞命中對照表")
            if keywords:
                comparison_rows = []
                for kw in keywords:
                    orig_status = "✅ 命中" if kw in orig_found else "❌ 缺漏"
                    opt_status = "✅ 命中" if kw in opt_found else "❌ 缺漏"
                    if kw in orig_found and kw not in opt_found:
                        reason = "優化版本遺失"
                    elif kw not in orig_found and kw in opt_found:
                        reason = "優化版本補上"
                    elif kw not in orig_found and kw not in opt_found:
                        reason = "兩版本皆缺漏"
                    else:
                        reason = "兩版本皆命中"
                    comparison_rows.append({
                        "關鍵詞": kw,
                        "原始版本": orig_status,
                        "優化版本": opt_status,
                        "差異說明": reason
                    })

                st.table(pd.DataFrame(comparison_rows))

            st.markdown("---")

            # 變化分析
            st.markdown("#### 🔄 關鍵詞覆蓋變化分析")

            newly_covered = sorted(set(opt_found) - set(orig_found))
            newly_lost = sorted(set(orig_found) - set(opt_found))
            remained_covered = sorted(set(orig_found) & set(opt_found))
            remained_missing = sorted(set(orig_missing) & set(opt_missing))

            change_col1, change_col2 = st.columns(2)

            with change_col1:
                st.markdown("**🆕 新增覆蓋**")
                if newly_covered:
                    for kw in newly_covered:
                        st.success(f"• {kw}")
                    st.success(f"🎉 新增覆蓋 {len(newly_covered)} 個關鍵詞！")
                else:
                    st.info("無新增覆蓋關鍵詞")

                st.markdown("**➡️ 持續覆蓋**")
                if remained_covered:
                    st.write(f"持續保持覆蓋 {len(remained_covered)} 個關鍵詞")
                    for kw in remained_covered:
                        st.write(f"  • {kw}")
                else:
                    st.info("無持續覆蓋的關鍵詞")

            with change_col2:
                st.markdown("**📉 失去覆蓋**")
                if newly_lost:
                    for kw in newly_lost:
                        st.error(f"• {kw}")
                    st.error(f"⚠️ 失去了 {len(newly_lost)} 個關鍵詞覆蓋！")
                else:
                    st.info("無失去覆蓋關鍵詞")

                st.markdown("**❌ 持續缺漏**")
                if remained_missing:
                    st.write(f"仍未覆蓋 {len(remained_missing)} 個關鍵詞")
                    for kw in remained_missing:
                        st.write(f"  • {kw}")
                else:
                    st.success("無持續缺漏的關鍵詞")
            
            # 結論和建議
            st.markdown("---")
            st.markdown("#### 💡 結論和建議")
            
            improvement_points = []
            
            if change > 5:
                st.success(f"🎉 **優秀表現**: 本題關鍵詞覆蓋率大幅改善 {change:.1f}%！")
            elif change > 0:
                st.success(f"✅ **正向改善**: 本題關鍵詞覆蓋率提升 {change:.1f}%")
            elif change == 0:
                st.info("➡️ **維持現狀**: 本題關鍵詞覆蓋率無變化")
            else:
                st.warning(f"⚠️ **需要改進**: 本題關鍵詞覆蓋率下降 {abs(change):.1f}%")
            
            # 具體建議
            if opt_missing:
                improvement_points.append(f"需要在回答中加入：{'、'.join(opt_missing)}")
            
            if newly_lost:
                improvement_points.append(f"避免遺失這些重要詞彙：{'、'.join(newly_lost)}")
            
            if newly_covered:
                improvement_points.append(f"繼續保持這些新增的優點：{'、'.join(newly_covered)}")
            
            if improvement_points:
                st.markdown("**📝 具體建議**")
                for point in improvement_points:
                    st.markdown(f"- {point}")
            else:
                st.success("🎆 本題關鍵詞覆蓋率已達到理想狀態！")
        
        else:
            st.warning("😔 無法載入資料，請先在「評估總覽」分頁中完成評估")

    with tab7:
        st.markdown("### 📥 下載結果")
        st.info("匯出完整評估報告（包含 GPT 人工評審結果）")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 完整評估報告（Excel）")

            if st.button("生成完整報告", type="primary"):
                # 準備包含 GPT 評分的完整資料
                export_df = results_df.copy()

                # 加入 GPT 詳細評分
                gpt_columns = ['GPT_RELEVANCE', 'GPT_COMPLETENESS', 'GPT_ACCURACY', 'GPT_FAITHFULNESS']
                for col in gpt_columns:
                    export_df[f'{col}_ORIGINAL'] = 0
                    export_df[f'{col}_OPTIMIZED'] = 0

                for idx, row in export_df.iterrows():
                    question_id = int(row['序號'])

                    if question_id in st.session_state.gpt_responses_original:
                        gpt_data = st.session_state.gpt_responses_original[question_id]
                        export_df.at[idx, 'GPT_RELEVANCE_ORIGINAL'] = gpt_data.get('relevance', 0)
                        export_df.at[idx, 'GPT_COMPLETENESS_ORIGINAL'] = gpt_data.get('completeness', 0)
                        export_df.at[idx, 'GPT_ACCURACY_ORIGINAL'] = gpt_data.get('accuracy', 0)
                        export_df.at[idx, 'GPT_FAITHFULNESS_ORIGINAL'] = gpt_data.get('faithfulness', 0)

                    if question_id in st.session_state.gpt_responses_optimized:
                        gpt_data = st.session_state.gpt_responses_optimized[question_id]
                        export_df.at[idx, 'GPT_RELEVANCE_OPTIMIZED'] = gpt_data.get('relevance', 0)
                        export_df.at[idx, 'GPT_COMPLETENESS_OPTIMIZED'] = gpt_data.get('completeness', 0)
                        export_df.at[idx, 'GPT_ACCURACY_OPTIMIZED'] = gpt_data.get('accuracy', 0)
                        export_df.at[idx, 'GPT_FAITHFULNESS_OPTIMIZED'] = gpt_data.get('faithfulness', 0)

                filename = f'RAG完整評估_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

                with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='評估結果', index=False)

                with open(filename, 'rb') as f:
                    st.download_button(
                        label="📥 下載完整報告",
                        data=f,
                        file_name=filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                if os.path.exists(filename):
                    os.remove(filename)

                st.success("✅ 完整報告已生成")

        with col2:
            st.markdown("#### 📈 GPT 評分摘要（JSON）")

            if st.button("匯出 GPT 評分", type="secondary"):
                gpt_export = {
                    "original": st.session_state.gpt_responses_original,
                    "optimized": st.session_state.gpt_responses_optimized,
                    "metadata": {
                        "total_questions": len(results_df),
                        "evaluated_original": len(st.session_state.gpt_responses_original),
                        "evaluated_optimized": len(st.session_state.gpt_responses_optimized),
                        "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }

                json_filename = f'GPT評分摘要_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(gpt_export, f, ensure_ascii=False, indent=2)

                with open(json_filename, 'rb') as f:
                    st.download_button(
                        label="📥 下載 GPT 評分",
                        data=f,
                        file_name=json_filename,
                        mime='application/json'
                    )

                if os.path.exists(json_filename):
                    os.remove(json_filename)

                st.success("✅ GPT 評分已匯出")

    with tab8:
        st.markdown("### 📝 GPT 評分補充說明")

        supplement_path = Path(__file__).resolve().parent / "GPT補充說明.md"

        if supplement_path.exists():
            try:
                supplement_content = supplement_path.read_text(encoding="utf-8")
                st.markdown(supplement_content)
            except Exception as exc:
                st.error(f"無法讀取 GPT補充說明.md：{exc}")
        else:
            st.warning("找不到 GPT補充說明.md，請確認檔案位於專案根目錄。")

else:
    # 未上傳檔案時的提示
    st.info("👈 請從側邊欄上傳測試結果檔案開始評估")

    # 使用說明
    with st.expander("📖 使用說明 v2.0 - 整合人工 GPT 評審", expanded=True):
        st.markdown("""
        ### 🎯 系統特性

        本系統採用**三種評估架構 + 人工 GPT 評審**，完美平衡自動化與準確度：

        #### 📊 評估流程

        1. **第一種：關鍵詞匹配**
           - 系統自動計算關鍵詞覆蓋率

        2. **第二種：語義相似度**
           - 系統自動計算語義相似度

        3. **第三種：GPT as a Judge**
           - （一）系統生成標準化 prompt
           - （二）您複製 prompt 到 ChatGPT
           - （三）貼回 ChatGPT 的 JSON 回應
           - （四）系統自動整合並即時更新所有指標

        #### 🚀 使用步驟

        1. 上傳測試結果檔案
        2. 系統自動完成關鍵詞和語義評估
        3. 進入「GPT 人工評審」頁籤
        4. 複製 prompt → 貼到 ChatGPT → 貼回結果
        5. 在「評估總覽」查看整合後的完整結果
        """)

# 頁尾
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>RAG 評估儀表板 v2.0 - 整合人工 GPT 評審</p>
    <p>© 2025 | 關鍵詞 + 語義相似度 + GPT 人工評審 | </p>
</div>
""", unsafe_allow_html=True)
