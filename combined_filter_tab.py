"""Combined metric filter tab for Streamlit dashboard."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd
import streamlit as st

DEFAULT_GPT_DIMS = ['relevance', 'completeness', 'accuracy', 'faithfulness']


def _score_range_slider(label: str, default_min: float = 0.0, default_max: float = 100.0, key: str | None = None) -> tuple[float, float]:
    return st.slider(
        label,
        min_value=0.0,
        max_value=100.0,
        value=(default_min, default_max),
        step=1.0,
        key=key,
    )


def _compute_per_question_scores(
    judge_df: pd.DataFrame,
    version_label: str,
    selected_dims: Iterable[str],
    dim_weights: Dict[str, float],
) -> Dict[int, float]:
    if judge_df.empty:
        return {}

    subset = judge_df[judge_df['version'].astype(str).str.lower() == version_label.lower()]
    if subset.empty:
        return {}

    scores: Dict[int, float] = {}
    for qid, group in subset.groupby('question_id'):
        if pd.isna(qid):
            continue
        score_sum = 0.0
        weight_sum = 0.0
        for dim in selected_dims:
            dim_rows = group[group['dimension'] == dim]
            if dim_rows.empty:
                continue
            score_val = pd.to_numeric(dim_rows.iloc[0].get('score'), errors='coerce')
            if pd.isna(score_val):
                continue
            weight = dim_weights.get(dim, 0.0)
            score_sum += float(score_val) * weight
            weight_sum += weight
        if weight_sum > 0:
            scores[int(qid)] = score_sum / weight_sum
    return scores


def render_combined_filter_tab(
    results_df: pd.DataFrame | None,
    enable_semantic: bool,
    enable_manual_gpt: bool,
    history_manager=None,
    selected_dims: Iterable[str] | None = None,
    dim_weights: Dict[str, float] | None = None,
) -> None:
    """Render tab for filtering questions by keyword, semantic, and GPT metrics."""
    st.markdown("### 🔎 綜合篩選器")
    st.info("依三種評分指標設定分數範圍，快速找出需要關注的題目。")

    if results_df is None or results_df.empty:
        st.warning("尚未載入測試結果，請先完成匯入。")
        return

    filter_df = results_df.copy()

    judge_df = None
    if history_manager is not None:
        try:
            judge_df = history_manager.load_llm_judge_table()
            if judge_df is not None:
                judge_df = judge_df.copy()
                judge_df['question_id'] = pd.to_numeric(judge_df.get('question_id'), errors='coerce')
        except Exception:
            judge_df = None

    selected_dims = list(selected_dims or DEFAULT_GPT_DIMS)
    if not selected_dims:
        selected_dims = DEFAULT_GPT_DIMS.copy()
    dim_weights = dim_weights or {dim: 1.0 / len(selected_dims) for dim in selected_dims}

    if enable_manual_gpt and judge_df is not None and not judge_df.empty:
        orig_scores = _compute_per_question_scores(judge_df, 'original', selected_dims, dim_weights)
        opt_scores = _compute_per_question_scores(judge_df, 'optimized', selected_dims, dim_weights)

        for idx, row in filter_df.iterrows():
            qid = int(row.get('序號', 0))
            if qid in orig_scores:
                filter_df.at[idx, 'GPT_OVERALL_ORIGINAL'] = orig_scores[qid]
            if qid in opt_scores:
                filter_df.at[idx, 'GPT_OVERALL_OPTIMIZED'] = opt_scores[qid]

    st.markdown("#### 🎯 篩選條件")
    col_kw, col_sem, col_gpt = st.columns(3)

    with col_kw:
        kw_min, kw_max = _score_range_slider("優化版關鍵詞覆蓋率", key="combined_filter_kw")
    with col_sem:
        if enable_semantic and "SEMANTIC_SIMILARITY_OPTIMIZED" in filter_df.columns:
            sem_min, sem_max = _score_range_slider("優化版語義相似度", key="combined_filter_sem")
        else:
            sem_min, sem_max = (0.0, 100.0)
            st.caption("語義相似度未啟用或資料缺失。")
    with col_gpt:
        gpt_available = enable_manual_gpt and "GPT_OVERALL_OPTIMIZED" in filter_df.columns
        if gpt_available:
            gpt_min, gpt_max = _score_range_slider("優化版 GPT 評分", key="combined_filter_gpt")
        else:
            gpt_min, gpt_max = (0.0, 100.0)
            st.caption("GPT 評分尚未填入。")

    mask = filter_df["KEYWORD_COVERAGE_OPTIMIZED"].between(kw_min, kw_max)
    if enable_semantic and "SEMANTIC_SIMILARITY_OPTIMIZED" in filter_df.columns:
        mask &= filter_df["SEMANTIC_SIMILARITY_OPTIMIZED"].between(sem_min, sem_max)
    if gpt_available:
        mask &= filter_df["GPT_OVERALL_OPTIMIZED"].between(gpt_min, gpt_max)

    filtered = filter_df[mask]

    st.markdown("#### 📋 篩選結果")
    st.metric("符合條件題數", len(filtered))

    if filtered.empty:
        st.info("沒有符合所選篩選條件的題目。")
        return

    columns: list[str] = [
        "序號",
        "測試問題",
        "KEYWORD_COVERAGE_ORIGINAL",
        "KEYWORD_COVERAGE_OPTIMIZED",
    ]
    rename_map: dict[str, str] = {
        "序號": "序號",
        "測試問題": "測試問題",
        "KEYWORD_COVERAGE_ORIGINAL": "原始-關鍵詞%",
        "KEYWORD_COVERAGE_OPTIMIZED": "優化-關鍵詞%",
    }

    if enable_semantic and "SEMANTIC_SIMILARITY_OPTIMIZED" in filtered.columns:
        columns.extend(["SEMANTIC_SIMILARITY_ORIGINAL", "SEMANTIC_SIMILARITY_OPTIMIZED"])
        rename_map.update({
            "SEMANTIC_SIMILARITY_ORIGINAL": "原始-語義%",
            "SEMANTIC_SIMILARITY_OPTIMIZED": "優化-語義%",
        })

    if gpt_available:
        columns.extend(["GPT_OVERALL_ORIGINAL", "GPT_OVERALL_OPTIMIZED"])
        rename_map.update({
            "GPT_OVERALL_ORIGINAL": "原始-GPT",
            "GPT_OVERALL_OPTIMIZED": "優化-GPT",
        })

    display_df = filtered[columns].rename(columns=rename_map)
    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "下載篩選結果 CSV",
        data=display_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="combined_filter_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
