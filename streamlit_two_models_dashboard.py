import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from rag_evaluation_two_models import RAGEvaluatorTwoModels
import os
from datetime import datetime
import json

st.set_page_config(
    page_title="RAG兩版本比較儀表板",
    page_icon="🆚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session state
if 'user_scores' not in st.session_state:
    st.session_state.user_scores = {}
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0

# 標題和說明
st.title("🆚 RAG 原始版本 vs 彙整版本 比較儀表板")
st.markdown("### 透過資料彙整優化提升UPGPT理解能力")

# 側邊欄
with st.sidebar:
    st.header("📁 設定與檔案選擇")
    
    # 選擇知識庫類型
    model_type = st.radio(
        "選擇比較模式",
        ["cross", "vector", "smart_doc"],
        format_func=lambda x: {
            "cross": "跨技術比較（向量原始 vs 智慧文檔彙整）",
            "vector": "向量知識庫（原始 vs 彙整）", 
            "smart_doc": "智慧文檔知識庫（原始 vs 彙整）"
        }.get(x, x),
        help="選擇要進行的比較類型",
        index=0  # 預設選擇跨技術比較
    )
    
    # 檔案選擇方式
    file_source = st.radio(
        "選擇檔案來源",
        ["📂 本地資料夾", "📤 上傳檔案"]
    )
    
    selected_file_path = None
    
    if file_source == "📂 本地資料夾":
        data_folder = "./test_data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        
        excel_files = [f for f in os.listdir(data_folder) 
                      if f.endswith(('.xlsx', '.xls')) and not f.startswith('~')]
        
        if excel_files:
            selected_file = st.selectbox(
                "選擇測試檔案",
                excel_files
            )
            selected_file_path = os.path.join(data_folder, selected_file)
        else:
            st.warning("⚠️ test_data 資料夾中沒有找到 Excel 檔案")
    else:
        uploaded_file = st.file_uploader(
            "上傳測試結果Excel檔案",
            type=['xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            selected_file_path = "temp_uploaded.xlsx"
            with open(selected_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    # 執行評估按鈕
    if selected_file_path:
        # 先顯示檔案資訊
        if st.checkbox("預覽檔案欄位", value=False):
            try:
                preview_df = pd.read_excel(selected_file_path)
                st.write("檔案包含的欄位:")
                st.write(list(preview_df.columns))
            except Exception as e:
                st.error(f"無法讀取檔案: {e}")
        
        if st.button("🚀 執行評估", type="primary", use_container_width=True):
            try:
                with st.spinner(f"正在評估{('向量知識庫' if model_type == 'vector' else '智慧文檔知識庫')}..."):
                    evaluator = RAGEvaluatorTwoModels(selected_file_path, model_type=model_type)
                    results = evaluator.evaluate_all()
                    stats = evaluator.generate_summary_stats()
                    
                    # 保存結果到session state
                    st.session_state['results'] = results
                    st.session_state['stats'] = stats
                    st.session_state['evaluator'] = evaluator
                    st.session_state['model_type'] = model_type
                    st.session_state['model_name'] = evaluator.model_name
                
                st.success("✅ 評估完成！")
                
                # 顯示使用的欄位
                st.info(f"使用的欄位對比: {evaluator.original_col} vs {evaluator.optimized_col}")
                
            except ValueError as e:
                st.error(f"❌ 錯誤: {e}")
                st.markdown("""
                ### 🔧 解決方法：
                1. 確認Excel檔案包含正確的欄位名稱
                2. 檢查是否選擇了正確的知識庫類型（向量/智慧文檔）
                3. 使用「預覽檔案欄位」功能查看實際的欄位名稱
                
                ### 📋 預期的欄位名稱：
                **跨技術比較:**
                - 原始版：向量知識庫（原始版）
                - 彙整版：智慧文檔知識庫（彙整版）
                
                **向量知識庫:**
                - 原始版：向量知識庫（原始版）
                - 彙整版：向量知識庫（彙整版）
                
                **智慧文檔知識庫:**
                - 原始版：智慧文檔知識庫（原始版）
                - 彙整版：智慧文檔知識庫（彙整版）
                """)
            except Exception as e:
                st.error(f"❌ 發生錯誤: {e}")
                import traceback
                st.text(traceback.format_exc())
    
    # 評分權重設定
    st.markdown("### ⚙️ 評分權重設定")
    coverage_weight = st.slider(
        "覆蓋率權重",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    faithfulness_weight = 1.0 - coverage_weight
    st.info(f"忠誠度權重: {faithfulness_weight:.1f}")

# 主要內容區
if 'results' in st.session_state:
    results = st.session_state['results']
    stats = st.session_state['stats']
    evaluator = st.session_state['evaluator']
    model_name = st.session_state['model_name']
    
    # 建立頁籤
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 總覽", "🎯 互動評分", "📈 改善分析", "🔍 問題層級", "💡 關鍵發現", "💾 下載結果"]
    )
    
    with tab1:
        st.header("評估總覽")
        st.info(f"正在評估：**{model_name}** | 資料筆數：**{len(results)}** 筆")
        
        # 關鍵指標卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            orig_coverage = stats['原始版本']['平均覆蓋率']
            opt_coverage = stats['彙整優化版本']['平均覆蓋率']
            coverage_lift = ((opt_coverage - orig_coverage) / orig_coverage * 100) if orig_coverage > 0 else 0
            
            st.metric(
                "覆蓋率提升",
                f"{opt_coverage:.1f}%",
                f"+{coverage_lift:.1f}%",
                help="彙整版本相比原始版本的覆蓋率"
            )
        
        with col2:
            orig_faith = stats['原始版本']['平均忠誠度']
            opt_faith = stats['彙整優化版本']['平均忠誠度']
            faith_lift = ((opt_faith - orig_faith) / orig_faith * 100) if orig_faith > 0 else 0
            
            st.metric(
                "忠誠度變化",
                f"{opt_faith:.1f}%",
                f"{faith_lift:+.1f}%",
                help="彙整版本相比原始版本的忠誠度"
            )
        
        with col3:
            improvement_rate = stats['改善效果']['顯著改善比例']
            
            st.metric(
                "顯著改善率",
                f"{improvement_rate:.1f}%",
                f"{int(improvement_rate * len(results) / 100)} 題",
                help="總體改善≥10%的題目比例"
            )
        
        with col4:
            regression_rate = stats['改善效果']['效果退步比例']
            
            st.metric(
                "需關注比例",
                f"{regression_rate:.1f}%",
                f"{int(regression_rate * len(results) / 100)} 題",
                help="效果退步的題目比例",
                delta_color="inverse"
            )
        
        # 比較表格
        st.markdown("### 📊 詳細指標對比")
        
        comparison_data = {
            '評估版本': ['🔴 原始版本', '🟢 彙整優化版本', '📈 改善幅度'],
            '平均覆蓋率': [
                f"{stats['原始版本']['平均覆蓋率']:.1f}%",
                f"{stats['彙整優化版本']['平均覆蓋率']:.1f}%",
                f"+{stats['改善效果']['平均覆蓋率提升']:.1f}%"
            ],
            '平均忠誠度': [
                f"{stats['原始版本']['平均忠誠度']:.1f}%",
                f"{stats['彙整優化版本']['平均忠誠度']:.1f}%",
                f"{stats['改善效果']['平均忠誠度提升']:+.1f}%"
            ],
            '平均綜合評分': [
                f"{stats['原始版本']['平均綜合評分']:.1f}%",
                f"{stats['彙整優化版本']['平均綜合評分']:.1f}%",
                f"+{stats['改善效果']['平均綜合評分提升']:.1f}%"
            ],
            '高覆蓋率比例': [
                f"{stats['原始版本']['高覆蓋率比例']:.1f}%",
                f"{stats['彙整優化版本']['高覆蓋率比例']:.1f}%",
                "-"
            ],
            '完全忠實比例': [
                f"{stats['原始版本']['完全忠實比例']:.1f}%",
                f"{stats['彙整優化版本']['完全忠實比例']:.1f}%",
                "-"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 使用樣式突出顯示
        def highlight_improvement(val):
            if isinstance(val, str) and val.startswith('+') and val != '+0.0%':
                return 'background-color: #2ECC71'
            elif isinstance(val, str) and val.startswith('-') and val != '-' and float(val[:-1]) < 0:
                return 'background-color: #E74C3C'
            return ''
        
        st.dataframe(
            comparison_df.style.applymap(highlight_improvement),
            use_container_width=True,
            hide_index=True
        )
        
        # 視覺化對比
        col_left, col_right = st.columns(2)
        
        with col_left:
            # 覆蓋率對比
            fig_coverage = go.Figure()
            fig_coverage.add_trace(go.Bar(
                x=['原始版本', '彙整優化版本'],
                y=[stats['原始版本']['平均覆蓋率'], stats['彙整優化版本']['平均覆蓋率']],
                text=[f"{stats['原始版本']['平均覆蓋率']:.1f}%", 
                      f"{stats['彙整優化版本']['平均覆蓋率']:.1f}%"],
                textposition='auto',
                marker_color=['#E74C3C', '#2ECC71']
            ))
            fig_coverage.update_layout(
                title="覆蓋率對比",
                yaxis_title="覆蓋率 (%)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_coverage, use_container_width=True)
        
        with col_right:
            # 忠誠度對比
            fig_faith = go.Figure()
            fig_faith.add_trace(go.Bar(
                x=['原始版本', '彙整優化版本'],
                y=[stats['原始版本']['平均忠誠度'], stats['彙整優化版本']['平均忠誠度']],
                text=[f"{stats['原始版本']['平均忠誠度']:.1f}%", 
                      f"{stats['彙整優化版本']['平均忠誠度']:.1f}%"],
                textposition='auto',
                marker_color=['#E74C3C', '#2ECC71']
            ))
            fig_faith.update_layout(
                title="忠誠度對比",
                yaxis_title="忠誠度 (%)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_faith, use_container_width=True)
    
    with tab2:
        st.header("🎯 互動式評分")
        st.markdown("比較原始版本與彙整版本的回答，並給予您的評分")
        
        # 問題導航
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.button("⬅️ 上一題", disabled=st.session_state.current_question_idx == 0):
                st.session_state.current_question_idx -= 1
        
        with col2:
            current_idx = st.selectbox(
                "選擇問題",
                range(len(results)),
                index=st.session_state.current_question_idx,
                format_func=lambda x: f"問題 {results.iloc[x]['序號']}: {results.iloc[x]['測試問題'][:50]}..."
            )
            st.session_state.current_question_idx = current_idx
        
        with col3:
            if st.button("下一題 ➡️", disabled=st.session_state.current_question_idx >= len(results) - 1):
                st.session_state.current_question_idx += 1
        
        # 顯示當前問題
        current_question = results.iloc[current_idx]
        
        st.markdown("### 📝 測試問題")
        st.info(current_question['測試問題'])
        
        # 顯示應回答詞彙
        with st.expander("查看應回答之詞彙", expanded=False):
            st.write(current_question['應回答之詞彙'])
        
        # 並排顯示兩個版本
        col_original, col_optimized = st.columns(2)
        
        with col_original:
            st.markdown("#### 🔴 原始版本")
            
            # AI評分
            st.markdown(f"**覆蓋率**: {current_question['SCORE_ORIGINAL']:.1f}%")
            st.markdown(f"**忠誠度**: {current_question['FAITHFULNESS_ORIGINAL']:.1f}%")
            st.markdown(f"**綜合評分**: {current_question['TOTAL_SCORE_ORIGINAL']:.1f}%")
            
            # 匹配的關鍵詞
            with st.expander("匹配的關鍵詞", expanded=False):
                st.write(current_question['MATCHED_KEYWORDS_ORIGINAL'])
            
            # 回答內容
            with st.expander("查看完整回答", expanded=True):
                st.write(current_question['ANSWER_ORIGINAL'])
            
            # 人工評分
            st.markdown("##### 您的評分")
            user_score_original = st.slider(
                "整體品質",
                min_value=1,
                max_value=5,
                value=3,
                key=f"score_orig_{current_idx}"
            )
        
        with col_optimized:
            st.markdown("#### 🟢 彙整優化版本")
            
            # AI評分和改善
            coverage_imp = current_question['COVERAGE_IMPROVEMENT']
            faith_imp = current_question['FAITHFULNESS_IMPROVEMENT']
            total_imp = current_question['TOTAL_IMPROVEMENT']
            
            st.markdown(f"**覆蓋率**: {current_question['SCORE_OPTIMIZED']:.1f}% {f'(+{coverage_imp:.1f}%)' if coverage_imp > 0 else f'({coverage_imp:.1f}%)'}")
            st.markdown(f"**忠誠度**: {current_question['FAITHFULNESS_OPTIMIZED']:.1f}% {f'(+{faith_imp:.1f}%)' if faith_imp > 0 else f'({faith_imp:.1f}%)'}")
            st.markdown(f"**綜合評分**: {current_question['TOTAL_SCORE_OPTIMIZED']:.1f}% {f'(+{total_imp:.1f}%)' if total_imp > 0 else f'({total_imp:.1f}%)'}")
            
            # 匹配的關鍵詞
            with st.expander("匹配的關鍵詞", expanded=False):
                st.write(current_question['MATCHED_KEYWORDS_OPTIMIZED'])
            
            # 回答內容
            with st.expander("查看完整回答", expanded=True):
                st.write(current_question['ANSWER_OPTIMIZED'])
            
            # 人工評分
            st.markdown("##### 您的評分")
            user_score_optimized = st.slider(
                "整體品質",
                min_value=1,
                max_value=5,
                value=3,
                key=f"score_opt_{current_idx}"
            )
        
        # 評語區域
        user_comment = st.text_area(
            "評語（選填）",
            placeholder="請分享您對兩個版本比較的看法...",
            key=f"comment_{current_idx}"
        )
        
        # 儲存按鈕
        if st.button("💾 儲存評分", type="primary"):
            question_id = f"q_{current_idx}"
            st.session_state.user_scores[question_id] = {
                'question_idx': current_idx,
                'question': current_question['測試問題'],
                'original_score': user_score_original,
                'optimized_score': user_score_optimized,
                'improvement': user_score_optimized - user_score_original,
                'comment': user_comment,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            st.success(f"✅ 已儲存問題 {current_question['序號']} 的評分")
        
        # 進度顯示
        scored_count = len(st.session_state.user_scores)
        progress = scored_count / len(results)
        st.progress(progress)
        st.info(f"評分進度: {scored_count}/{len(results)} ({progress*100:.1f}%)")
    
    with tab3:
        st.header("📈 改善分析")
        
        # 改善分布圖
        fig_improvement = px.histogram(
            results,
            x='TOTAL_IMPROVEMENT',
            nbins=20,
            title="總體改善分布",
            labels={'TOTAL_IMPROVEMENT': '改善幅度 (%)'},
            color_discrete_sequence=['#3498DB']
        )
        
        # 添加平均線
        avg_improvement = results['TOTAL_IMPROVEMENT'].mean()
        fig_improvement.add_vline(
            x=avg_improvement,
            line_dash="dash",
            line_color="red",
            annotation_text=f"平均: {avg_improvement:.1f}%"
        )
        
        fig_improvement.add_vline(
            x=0,
            line_color="gray",
            annotation_text="無變化"
        )
        
        st.plotly_chart(fig_improvement, use_container_width=True)
        
        # 改善相關性分析
        fig_scatter = px.scatter(
            results,
            x='COVERAGE_IMPROVEMENT',
            y='FAITHFULNESS_IMPROVEMENT',
            color='TOTAL_IMPROVEMENT',
            size=abs(results['TOTAL_IMPROVEMENT']),
            hover_data=['序號', '測試問題'],
            title="覆蓋率改善 vs 忠誠度改善",
            labels={
                'COVERAGE_IMPROVEMENT': '覆蓋率改善 (%)',
                'FAITHFULNESS_IMPROVEMENT': '忠誠度改善 (%)',
                'TOTAL_IMPROVEMENT': '總體改善 (%)'
            },
            color_continuous_scale='RdYlGn'
        )
        
        # 添加象限
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 改善分類
        st.markdown("### 📊 改善效果分類")
        
        improvement_categories = []
        
        significant_improve = (results['TOTAL_IMPROVEMENT'] >= 10).sum()
        moderate_improve = ((results['TOTAL_IMPROVEMENT'] > 0) & (results['TOTAL_IMPROVEMENT'] < 10)).sum()
        no_change = (results['TOTAL_IMPROVEMENT'] == 0).sum()
        regression = (results['TOTAL_IMPROVEMENT'] < 0).sum()
        
        if significant_improve > 0:
            improvement_categories.append({'類別': '顯著改善 (≥10%)', '數量': significant_improve})
        if moderate_improve > 0:
            improvement_categories.append({'類別': '略有改善 (0-10%)', '數量': moderate_improve})
        if no_change > 0:
            improvement_categories.append({'類別': '無變化', '數量': no_change})
        if regression > 0:
            improvement_categories.append({'類別': '效果退步', '數量': regression})
        
        if improvement_categories:
            cat_df = pd.DataFrame(improvement_categories)
            
            fig_pie = px.pie(
                cat_df,
                values='數量',
                names='類別',
                title="改善效果分布",
                color_discrete_map={
                    '顯著改善 (≥10%)': '#2ECC71',
                    '略有改善 (0-10%)': '#F1C40F',
                    '無變化': '#95A5A6',
                    '效果退步': '#E74C3C'
                }
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab4:
        st.header("🔍 問題層級分析")
        
        # 篩選選項
        filter_option = st.selectbox(
            "篩選顯示",
            ["所有問題", "顯著改善 (≥10%)", "略有改善", "無變化", "效果退步", "已評分問題"]
        )
        
        # 根據條件篩選
        if filter_option == "顯著改善 (≥10%)":
            filtered_results = results[results['TOTAL_IMPROVEMENT'] >= 10]
        elif filter_option == "略有改善":
            filtered_results = results[(results['TOTAL_IMPROVEMENT'] > 0) & (results['TOTAL_IMPROVEMENT'] < 10)]
        elif filter_option == "無變化":
            filtered_results = results[results['TOTAL_IMPROVEMENT'] == 0]
        elif filter_option == "效果退步":
            filtered_results = results[results['TOTAL_IMPROVEMENT'] < 0]
        elif filter_option == "已評分問題":
            scored_indices = [int(key.split('_')[1]) for key in st.session_state.user_scores.keys()]
            filtered_results = results.iloc[scored_indices] if scored_indices else pd.DataFrame()
        else:
            filtered_results = results
        
        if not filtered_results.empty:
            # 顯示表格
            display_columns = [
                '序號', '測試問題',
                'SCORE_ORIGINAL', 'SCORE_OPTIMIZED', 'COVERAGE_IMPROVEMENT',
                'FAITHFULNESS_ORIGINAL', 'FAITHFULNESS_OPTIMIZED', 'FAITHFULNESS_IMPROVEMENT',
                'TOTAL_SCORE_ORIGINAL', 'TOTAL_SCORE_OPTIMIZED', 'TOTAL_IMPROVEMENT'
            ]
            
            display_df = filtered_results[display_columns].copy()
            display_df.columns = [
                '序號', '測試問題',
                '原始覆蓋率', '優化覆蓋率', '覆蓋率改善',
                '原始忠誠度', '優化忠誠度', '忠誠度改善',
                '原始綜合', '優化綜合', '總體改善'
            ]
            
            # 格式化數值
            for col in display_df.columns[2:]:
                display_df[col] = display_df[col].round(1)
            
            st.dataframe(display_df, use_container_width=True, height=500)
            
            # Top 5 改善最多
            st.markdown("### 🏆 改善最顯著的問題 (Top 5)")
            top_5 = filtered_results.nlargest(5, 'TOTAL_IMPROVEMENT')[['序號', '測試問題', 'TOTAL_IMPROVEMENT']]
            st.dataframe(top_5, use_container_width=True)
        else:
            st.warning("沒有符合條件的資料")
    
    with tab5:
        st.header("💡 關鍵發現與建議")
        
        # 計算關鍵統計
        total_questions = len(results)
        significant_improvements = (results['TOTAL_IMPROVEMENT'] >= 10).sum()
        coverage_improvements = (results['COVERAGE_IMPROVEMENT'] > 0).sum()
        faithfulness_changes = results['FAITHFULNESS_IMPROVEMENT']
        
        # 主要發現
        st.markdown("### 🔍 主要發現")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **覆蓋率改善情況**
            - 有 {coverage_improvements} 題 ({coverage_improvements/total_questions*100:.1f}%) 覆蓋率提升
            - 平均提升 {stats['改善效果']['平均覆蓋率提升']:.1f}%
            - 最大提升 {results['COVERAGE_IMPROVEMENT'].max():.1f}%
            """)
            
        with col2:
            faith_improved = (faithfulness_changes > 0).sum()
            faith_maintained = (faithfulness_changes == 0).sum()
            faith_decreased = (faithfulness_changes < 0).sum()
            
            st.info(f"""
            **忠誠度變化情況**
            - 提升: {faith_improved} 題 ({faith_improved/total_questions*100:.1f}%)
            - 維持: {faith_maintained} 題 ({faith_maintained/total_questions*100:.1f}%)
            - 下降: {faith_decreased} 題 ({faith_decreased/total_questions*100:.1f}%)
            """)
        
        # 優化效果總結
        st.markdown("### 📊 優化效果總結")
        
        if significant_improvements > total_questions * 0.5:
            st.success(f"""
            ✅ **優化效果顯著**
            - 超過一半的問題 ({significant_improvements}/{total_questions}) 達到顯著改善
            - 資料彙整策略有效提升了UPGPT的理解能力
            - 建議繼續使用彙整版本的知識庫
            """)
        elif significant_improvements > total_questions * 0.3:
            st.warning(f"""
            ⚠️ **優化效果中等**
            - 約 {significant_improvements/total_questions*100:.0f}% 的問題達到顯著改善
            - 資料彙整在部分場景下有效
            - 建議針對未改善的問題類型進一步優化
            """)
        else:
            st.error(f"""
            ❌ **優化效果有限**
            - 僅 {significant_improvements/total_questions*100:.0f}% 的問題達到顯著改善
            - 需要重新檢視資料彙整策略
            - 建議分析退步案例，調整優化方法
            """)
        
        # 具體建議
        st.markdown("### 💡 優化建議")
        
        # 找出退步最多的問題類型
        regression_questions = results[results['TOTAL_IMPROVEMENT'] < -5]
        
        if len(regression_questions) > 0:
            st.markdown("#### 需要關注的退步問題：")
            for idx, row in regression_questions.head(3).iterrows():
                st.markdown(f"- 問題 {row['序號']}: {row['測試問題'][:50]}... (退步 {abs(row['TOTAL_IMPROVEMENT']):.1f}%)")
        
        # 改善潛力分析
        low_coverage_original = results[results['SCORE_ORIGINAL'] < 50]
        if len(low_coverage_original) > 0:
            improved_count = (low_coverage_original['COVERAGE_IMPROVEMENT'] > 10).sum()
            st.markdown(f"""
            #### 低覆蓋率問題改善情況：
            - 原始版本中有 {len(low_coverage_original)} 題覆蓋率低於50%
            - 其中 {improved_count} 題在彙整後有顯著改善
            - 改善率: {improved_count/len(low_coverage_original)*100:.1f}%
            """)
    
    with tab6:
        st.header("💾 下載評估結果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 AI評估報告")
            
            if st.button("生成完整評估報告", type="primary"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'{model_name}_評估報告_{timestamp}.xlsx'
                
                # 使用evaluator的save_results方法
                evaluator.save_results(filename)
                
                # 提供下載
                with open(filename, 'rb') as f:
                    st.download_button(
                        label="📥 下載評估報告",
                        data=f,
                        file_name=filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                
                st.success(f"✅ 報告已生成：{filename}")
        
        with col2:
            st.markdown("### 👤 人工評分結果")
            
            if st.session_state.user_scores:
                if st.button("匯出人工評分", type="secondary"):
                    # 準備評分數據
                    scores_data = []
                    for key, value in st.session_state.user_scores.items():
                        scores_data.append({
                            '問題序號': results.iloc[value['question_idx']]['序號'],
                            '測試問題': value['question'],
                            '原始版本評分': value['original_score'],
                            '優化版本評分': value['optimized_score'],
                            '評分改善': value['improvement'],
                            '評語': value.get('comment', ''),
                            '評分時間': value['timestamp']
                        })
                    
                    scores_df = pd.DataFrame(scores_data)
                    
                    # 生成檔案
                    user_filename = f'{model_name}_人工評分_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                    scores_df.to_excel(user_filename, index=False)
                    
                    # 提供下載
                    with open(user_filename, 'rb') as f:
                        st.download_button(
                            label="📥 下載人工評分",
                            data=f,
                            file_name=user_filename,
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                        )
                    
                    st.success("✅ 人工評分已匯出")
            else:
                st.warning("尚未進行任何人工評分")
        
        # 顯示統計摘要
        st.markdown("### 📈 評估統計摘要")
        
        summary_data = []
        for category, metrics in stats.items():
            for metric, value in metrics.items():
                summary_data.append({
                    '類別': category,
                    '指標': metric,
                    '數值': f"{value:.2f}%"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

else:
    # 沒有上傳檔案時的提示
    st.info("👈 請從側邊欄選擇檔案並執行評估")
    
    # 顯示使用說明
    with st.expander("📖 使用說明", expanded=True):
        st.markdown("""
        ### 🎯 系統特色
        
        本系統專注於比較**原始版本**與**彙整優化版本**兩個知識庫的表現，
        展示透過資料彙整如何提升UPGPT的理解和回答能力。
        
        ### 📊 評估指標說明
        
        1. **覆蓋率** (Coverage)
           - 衡量回答中包含多少應回答的關鍵資訊
           - 越高表示回答越完整
        
        2. **忠誠度** (Faithfulness)
           - 衡量回答是否忠實於原始資料
           - 避免AI產生幻覺或虛構內容
        
        3. **綜合評分**
           - 覆蓋率和忠誠度的加權平均
           - 可在側邊欄調整權重
        
        ### 🔄 工作流程
        
        1. **選擇知識庫類型** - 向量知識庫或智慧文檔知識庫
        2. **上傳測試檔案** - 包含測試問題和兩個版本回答的Excel
        3. **執行評估** - 自動計算各項指標
        4. **查看結果** - 多維度分析改善效果
        5. **互動評分** - 提供人工評分和評語
        6. **下載報告** - 匯出完整評估結果
        
        ### 💡 優化價值
        
        透過資料彙整優化，可以：
        - 提高回答的完整性（覆蓋率）
        - 維持或改善回答的準確性（忠誠度）
        - 讓UPGPT更好地理解和組織知識
        """)

# 頁尾
st.markdown("---")
st.caption("RAG 兩版本比較系統 v1.0 | 專注展示資料彙整優化的價值")