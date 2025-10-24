"""
RAG 比較儀表板 v2.0 - 三層評估架構
====================================

升級特性：
1. 支援三層評估結果顯示（關鍵詞、語義、GPT）
2. 可配置評估層級和權重
3. 多維度數據視覺化
4. 成本預估和效能分析
5. 完整的資料輸入輸出支援

版本：2.0
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
from rag_evaluation_two_models_v2 import RAGEvaluatorV2

# 設定頁面配置
st.set_page_config(
    page_title="RAG 評估儀表板 v2.0 - 三層評估架構",
    page_icon="🆚",
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

# 標題和說明
st.title("🆚 RAG 評估儀表板 v2.0")
st.markdown("### 三層評估架構：關鍵詞 + 語義相似度 + GPT 評審")

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

    enable_gpt = st.checkbox(
        "啟用 GPT 評審",
        value=False,
        help="使用 GPT 進行多維度深度評估（需要 API 金鑰，會產生費用）"
    )

    openai_api_key = None
    if enable_gpt:
        openai_api_key = st.text_input(
            "OpenAI API 金鑰",
            type="password",
            help="請輸入您的 OpenAI API 金鑰"
        )

        if not openai_api_key:
            st.warning("⚠️ 請輸入 API 金鑰以啟用 GPT 評審")
            enable_gpt = False

    # 評分權重設定
    st.markdown("### ⚖️ 評分權重設定")

    if enable_semantic and enable_gpt:
        st.info("三層評估模式")
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
    elif enable_gpt:
        st.info("雙層評估模式（關鍵詞 + GPT）")
        weight_keyword = st.slider("關鍵詞權重", 0.0, 1.0, 0.4, 0.1)
        weight_gpt = 1.0 - weight_keyword
        weight_semantic = 0.0
        st.metric("GPT 權重", f"{weight_gpt:.1f}")
    else:
        st.info("單層評估模式（僅關鍵詞）")
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

# 主要內容區
if uploaded_file is not None:
    # 處理檔案
    if isinstance(uploaded_file, str):
        temp_file_path = uploaded_file
    else:
        temp_file_path = "temp_comparison_file.xlsx"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

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
            enable_gpt=enable_gpt,
            openai_api_key=openai_api_key,
            weights=weights
        )

        st.session_state.evaluator_instance = evaluator

        # 執行評估
        with st.spinner("🔄 正在進行三層評估分析..."):
            results_df = evaluator.evaluate_all()
            st.session_state.comparison_results = results_df

        # 清理臨時檔案
        if os.path.exists(temp_file_path) and not isinstance(uploaded_file, str):
            os.remove(temp_file_path)

    except Exception as e:
        st.error(f"❌ 評估過程中發生錯誤：{str(e)}")
        if os.path.exists(temp_file_path) and not isinstance(uploaded_file, str):
            os.remove(temp_file_path)
        st.stop()

    # 建立頁籤
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 評估總覽", "📈 詳細對比", "🔍 層級分析", "💬 問題導覽", "📥 下載結果"]
    )

    with tab1:
        st.markdown("### 📊 評估總覽")

        # 獲取統計數據
        stats = evaluator.generate_summary_stats()

        # 關鍵指標卡片
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**📈 綜合評分提升**")
            improvement = stats['改善效果']['平均綜合評分提升']
            color = '#28a745' if improvement > 0 else '#dc3545'
            st.markdown(f"<h1 style='color: {color}; margin: 0;'>{stats['彙整優化版本']['平均綜合評分']:.1f}分</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if improvement > 0 else '↓'} {abs(improvement):.1f}分</p>", unsafe_allow_html=True)

        with col2:
            st.markdown("**🎯 關鍵詞覆蓋率**")
            keyword_improvement = stats['改善效果']['平均關鍵詞覆蓋率提升']
            color = '#28a745' if keyword_improvement > 0 else '#dc3545'
            st.markdown(f"<h1 style='color: {color}; margin: 0;'>{stats['彙整優化版本']['平均關鍵詞覆蓋率']:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if keyword_improvement > 0 else '↓'} {abs(keyword_improvement):.1f}%</p>", unsafe_allow_html=True)

        with col3:
            if enable_semantic:
                st.markdown("**🔤 語義相似度**")
                semantic_improvement = stats['改善效果']['平均語義相似度提升']
                color = '#28a745' if semantic_improvement > 0 else '#dc3545'
                st.markdown(f"<h1 style='color: {color}; margin: 0;'>{stats['彙整優化版本']['平均語義相似度']:.1f}%</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if semantic_improvement > 0 else '↓'} {abs(semantic_improvement):.1f}%</p>", unsafe_allow_html=True)
            else:
                st.info("語義相似度未啟用")

        with col4:
            if enable_gpt:
                st.markdown("**🤖 GPT 評分**")
                gpt_improvement = stats['改善效果']['平均GPT評分提升']
                color = '#28a745' if gpt_improvement > 0 else '#dc3545'
                st.markdown(f"<h1 style='color: {color}; margin: 0;'>{stats['彙整優化版本']['平均GPT評分']:.1f}分</h1>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if gpt_improvement > 0 else '↓'} {abs(gpt_improvement):.1f}分</p>", unsafe_allow_html=True)
            else:
                st.info("GPT 評審未啟用")

        # 評估層級配置顯示
        st.markdown("### ⚙️ 評估配置")
        config_col1, config_col2, config_col3 = st.columns(3)

        with config_col1:
            st.metric("關鍵詞匹配", "✅ 啟用", f"權重: {weights['keyword']:.0%}")

        with config_col2:
            status = "✅ 啟用" if enable_semantic else "❌ 停用"
            st.metric("語義相似度", status, f"權重: {weights['semantic']:.0%}")

        with config_col3:
            status = "✅ 啟用" if enable_gpt else "❌ 停用"
            st.metric("GPT 評審", status, f"權重: {weights['gpt']:.0%}")

        # 改善統計
        st.markdown("### 📈 改善統計")

        stat_col1, stat_col2, stat_col3 = st.columns(3)

        with stat_col1:
            significant_improvements = (results_df['FINAL_IMPROVEMENT'] >= improvement_threshold).sum()
            improvement_rate = significant_improvements / len(results_df) * 100
            st.metric(
                "顯著改善問題數",
                f"{significant_improvements} 題",
                f"{improvement_rate:.1f}%"
            )

        with stat_col2:
            no_change = (results_df['FINAL_IMPROVEMENT'] == 0).sum()
            st.metric("無變化問題數", f"{no_change} 題")

        with stat_col3:
            declined = (results_df['FINAL_IMPROVEMENT'] < 0).sum()
            declined_rate = declined / len(results_df) * 100
            st.metric(
                "退步問題數",
                f"{declined} 題",
                f"{declined_rate:.1f}%",
                delta_color="inverse"
            )

        # 分數分布圖表
        st.markdown("### 📊 綜合評分分布")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=results_df['FINAL_SCORE_ORIGINAL'],
            name='原始版本',
            opacity=0.7,
            marker_color='#e57373',
            nbinsx=20
        ))

        fig.add_trace(go.Histogram(
            x=results_df['FINAL_SCORE_OPTIMIZED'],
            name='優化版本',
            opacity=0.7,
            marker_color='#81c784',
            nbinsx=20
        ))

        fig.update_layout(
            barmode='overlay',
            title='綜合評分分布對比',
            xaxis_title='綜合評分',
            yaxis_title='問題數量',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 📈 詳細對比分析")

        # 建立多層級對比表格
        comparison_data = []

        metrics = []
        if True:  # 關鍵詞總是啟用
            metrics.append(('關鍵詞覆蓋率', 'KEYWORD_COVERAGE'))
        if enable_semantic:
            metrics.append(('語義相似度', 'SEMANTIC_SIMILARITY'))
        if enable_gpt:
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
                '最高分': f"{results_df[f'{metric_key.replace("SCORE", "IMPROVEMENT").replace("COVERAGE", "IMPROVEMENT").replace("SIMILARITY", "IMPROVEMENT").replace("OVERALL", "IMPROVEMENT")}'].max():+.1f}",
                '最低分': f"{results_df[f'{metric_key.replace("SCORE", "IMPROVEMENT").replace("COVERAGE", "IMPROVEMENT").replace("SIMILARITY", "IMPROVEMENT").replace("OVERALL", "IMPROVEMENT")}'].min():+.1f}",
                '標準差': "-"
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # 雷達圖對比
        st.markdown("### 🎯 多維度雷達圖對比")

        categories = []
        original_scores = []
        optimized_scores = []

        if True:  # 關鍵詞
            categories.append('關鍵詞覆蓋率')
            original_scores.append(results_df['KEYWORD_COVERAGE_ORIGINAL'].mean())
            optimized_scores.append(results_df['KEYWORD_COVERAGE_OPTIMIZED'].mean())

        if enable_semantic:
            categories.append('語義相似度')
            original_scores.append(results_df['SEMANTIC_SIMILARITY_ORIGINAL'].mean())
            optimized_scores.append(results_df['SEMANTIC_SIMILARITY_OPTIMIZED'].mean())

        if enable_gpt:
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

    with tab3:
        st.markdown("### 🔍 層級分析")
        st.info("深入分析各評估層級的貢獻度和改善效果")

        # 層級貢獻度分析
        st.markdown("#### 📊 各層級評分貢獻度")

        layer_col1, layer_col2 = st.columns(2)

        with layer_col1:
            # 原始版本貢獻度
            original_contributions = []
            labels = []

            if weight_keyword > 0:
                original_contributions.append(
                    results_df['KEYWORD_COVERAGE_ORIGINAL'].mean() * weight_keyword
                )
                labels.append(f'關鍵詞 ({weight_keyword:.0%})')

            if weight_semantic > 0:
                original_contributions.append(
                    results_df['SEMANTIC_SIMILARITY_ORIGINAL'].mean() * weight_semantic
                )
                labels.append(f'語義 ({weight_semantic:.0%})')

            if weight_gpt > 0:
                original_contributions.append(
                    results_df['GPT_OVERALL_ORIGINAL'].mean() * weight_gpt
                )
                labels.append(f'GPT ({weight_gpt:.0%})')

            fig_orig = go.Figure(data=[go.Pie(
                labels=labels,
                values=original_contributions,
                title='原始版本貢獻度'
            )])

            st.plotly_chart(fig_orig, use_container_width=True)

        with layer_col2:
            # 優化版本貢獻度
            optimized_contributions = []

            if weight_keyword > 0:
                optimized_contributions.append(
                    results_df['KEYWORD_COVERAGE_OPTIMIZED'].mean() * weight_keyword
                )

            if weight_semantic > 0:
                optimized_contributions.append(
                    results_df['SEMANTIC_SIMILARITY_OPTIMIZED'].mean() * weight_semantic
                )

            if weight_gpt > 0:
                optimized_contributions.append(
                    results_df['GPT_OVERALL_OPTIMIZED'].mean() * weight_gpt
                )

            fig_opt = go.Figure(data=[go.Pie(
                labels=labels,
                values=optimized_contributions,
                title='優化版本貢獻度'
            )])

            st.plotly_chart(fig_opt, use_container_width=True)

        # 改善分布分析
        st.markdown("#### 📈 各層級改善分布")

        improvement_cols = []
        improvement_names = []

        if True:  # 關鍵詞
            improvement_cols.append('KEYWORD_IMPROVEMENT')
            improvement_names.append('關鍵詞覆蓋率')

        if enable_semantic:
            improvement_cols.append('SEMANTIC_IMPROVEMENT')
            improvement_names.append('語義相似度')

        if enable_gpt:
            improvement_cols.append('GPT_IMPROVEMENT')
            improvement_names.append('GPT 評分')

        fig_improvements = make_subplots(
            rows=1,
            cols=len(improvement_cols),
            subplot_titles=improvement_names
        )

        for idx, (col, name) in enumerate(zip(improvement_cols, improvement_names), 1):
            fig_improvements.add_trace(
                go.Histogram(
                    x=results_df[col],
                    name=name,
                    nbinsx=20
                ),
                row=1,
                col=idx
            )

        fig_improvements.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_improvements, use_container_width=True)

    with tab4:
        st.markdown("### 💬 問題導覽")
        st.info("瀏覽所有測試問題的詳細評估結果")

        # 篩選選項
        filter_option = st.selectbox(
            "篩選顯示",
            ["所有問題", "顯著改善", "略有改善", "無變化", "效果退步"]
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
        else:
            filtered_df = results_df

        st.info(f"顯示 {len(filtered_df)} / {len(results_df)} 個問題")

        # 顯示問題列表
        for idx, row in filtered_df.iterrows():
            with st.expander(f"問題 {row['序號']}: {row['測試問題'][:50]}..."):
                # 問題資訊
                st.markdown(f"**測試問題**: {row['測試問題']}")
                st.markdown(f"**應回答詞彙**: {row['應回答之詞彙']}")

                # 評分對比
                score_col1, score_col2, score_col3 = st.columns(3)

                with score_col1:
                    st.metric(
                        "關鍵詞覆蓋率",
                        f"{row['KEYWORD_COVERAGE_OPTIMIZED']:.1f}%",
                        f"{row['KEYWORD_IMPROVEMENT']:.1f}%"
                    )

                with score_col2:
                    if enable_semantic:
                        st.metric(
                            "語義相似度",
                            f"{row['SEMANTIC_SIMILARITY_OPTIMIZED']:.1f}%",
                            f"{row['SEMANTIC_IMPROVEMENT']:.1f}%"
                        )

                with score_col3:
                    if enable_gpt:
                        st.metric(
                            "GPT 評分",
                            f"{row['GPT_OVERALL_OPTIMIZED']:.1f}",
                            f"{row['GPT_IMPROVEMENT']:.1f}"
                        )

                # 綜合評分
                st.metric(
                    "📊 綜合評分",
                    f"{row['FINAL_SCORE_OPTIMIZED']:.1f}",
                    f"{row['FINAL_IMPROVEMENT']:.1f}"
                )

                # GPT 推理（如果有��
                if enable_gpt and row['GPT_REASONING_OPTIMIZED']:
                    st.markdown("**🤖 GPT 評審意見**")
                    st.info(row['GPT_REASONING_OPTIMIZED'])

    with tab5:
        st.markdown("### 📥 下載結果")
        st.info("匯出完整評估報告")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 評估報告（Excel）")

            if st.button("生成評估報告", type="primary"):
                filename = f'RAG評估報告_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                evaluator.save_results(filename)

                with open(filename, 'rb') as f:
                    st.download_button(
                        label="📥 下載評估報告",
                        data=f,
                        file_name=filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                if os.path.exists(filename):
                    os.remove(filename)

                st.success("✅ 評估報告已生成")

        with col2:
            st.markdown("#### 📈 統計摘要（JSON）")

            if st.button("生成統計摘要", type="secondary"):
                stats = evaluator.generate_summary_stats()

                json_filename = f'統計摘要_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)

                with open(json_filename, 'rb') as f:
                    st.download_button(
                        label="📥 下載統計摘要",
                        data=f,
                        file_name=json_filename,
                        mime='application/json'
                    )

                if os.path.exists(json_filename):
                    os.remove(json_filename)

                st.success("✅ 統計摘要已生成")

else:
    # 未上傳檔案時的提示
    st.info("👈 請從側邊欄上傳測試結果檔案開始評估")

    # 使用說明
    with st.expander("📖 使用說明 v2.0", expanded=True):
        st.markdown("""
        ### 🎯 系統特性

        本系統採用**三層評估架構**，提供全方位的 RAG 系統品質評估：

        #### 📊 三層評估架構

        1. **第一層：關鍵詞匹配**（必選，快速）
           - 評估回答中包含的關鍵詞比例
           - 支援同義詞識別
           - 評估速度：極快

        2. **第二層：語義相似度**（可選，推薦）
           - 使用 Sentence Transformers 計算語義相似度
           - 捕捉語義層面的匹配度
           - 評估速度：中等
           - 需要：安裝 sentence-transformers

        3. **第三層：GPT as a Judge**（可選，深度）
           - 多維度評估：相關性、完整性、準確性、忠實度
           - 提供質化反饋和改進建議
           - 評估速度：較慢
           - 需要：OpenAI API 金鑰（會產生費用）

        #### 🚀 開始使用

        1. 選擇評估層級（建議啟用語義相似度）
        2. 上傳測試結果 Excel/CSV 檔案
        3. 設定評分權重（系統會自動建議）
        4. 查看多維度評估結果
        5. 導出完整報告

        #### 📝 檔案格式要求

        - **測試問題**: 測試的問題內容
        - **應回答詞彙**: 期望回答包含的關鍵詞彙
        - **向量知識庫（原始版）**: 原始版本的回答
        - **智慧文檔知識庫（彙整版）**: 優化版本的回答

        #### 💡 最佳實踐

        - **快速評估**: 僅使用關鍵詞匹配
        - **推薦配置**: 關鍵詞 + 語義相似度（平衡速度和準確度）
        - **深度分析**: 啟用所有三層評估（最準確，但較慢）

        #### 💰 成本估算（GPT 評審）

        - 約 $0.002 USD / 問題（使用 GPT-3.5-turbo）
        - 100 題測試集約 $0.20 USD
        """)

# 頁尾
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>RAG 評估儀表板 v2.0 - 三層評估架構</p>
    <p>© 2024 | 關鍵詞匹配 + 語義相似度 + GPT 評審</p>
</div>
""", unsafe_allow_html=True)
