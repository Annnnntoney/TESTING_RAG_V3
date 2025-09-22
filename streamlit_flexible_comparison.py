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
import jieba
import re
from typing import List, Dict, Tuple

st.set_page_config(
    page_title="RAG 彈性比較系統",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'user_scores' not in st.session_state:
    st.session_state.user_scores = {}
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0

# 標題
st.title("🔄 RAG 彈性比較系統")
st.markdown("### 自由選擇任意兩個欄位進行比較分析")

# 側邊欄
with st.sidebar:
    st.header("📁 檔案與欄位設定")
    
    # 檔案上傳
    uploaded_file = st.file_uploader(
        "上傳Excel檔案",
        type=['xlsx', 'xls'],
        help="請上傳包含測試結果的Excel檔案"
    )
    
    if uploaded_file:
        # 讀取檔案
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ 成功載入 {len(df)} 筆資料")
        
        # 顯示所有欄位
        st.markdown("### 📋 可用欄位")
        all_columns = df.columns.tolist()
        
        # 過濾出可能包含回答內容的欄位
        answer_columns = [col for col in all_columns if 
                         any(keyword in col for keyword in ['知識庫', '回答', 'answer', 'response', 'output'])]
        
        if not answer_columns:
            answer_columns = all_columns
        
        # 讓使用者選擇要比較的兩個欄位
        st.markdown("### 🎯 選擇比較欄位")
        
        col1 = st.selectbox(
            "選擇第一個欄位（基準版本）",
            answer_columns,
            index=0,
            help="選擇作為比較基準的欄位"
        )
        
        # 排除已選擇的第一個欄位
        available_for_col2 = [col for col in answer_columns if col != col1]
        
        col2 = st.selectbox(
            "選擇第二個欄位（比較版本）",
            available_for_col2,
            index=0 if available_for_col2 else None,
            help="選擇要與基準版本比較的欄位"
        )
        
        # 選擇必要欄位
        st.markdown("### 🔧 必要欄位設定")
        
        # 測試問題欄位
        question_col = st.selectbox(
            "測試問題欄位",
            all_columns,
            index=all_columns.index('測試問題') if '測試問題' in all_columns else 0
        )
        
        # 應回答詞彙欄位
        keywords_col = st.selectbox(
            "應回答詞彙欄位",
            all_columns,
            index=all_columns.index('應回答之詞彙') if '應回答之詞彙' in all_columns else 0
        )
        
        # 權重設定
        st.markdown("### ⚙️ 評分權重")
        coverage_weight = st.slider(
            "覆蓋率權重",
            0.0, 1.0, 0.5, 0.1
        )
        faithfulness_weight = 1.0 - coverage_weight
        st.info(f"忠誠度權重: {faithfulness_weight:.1f}")

# 主要內容區
if uploaded_file and 'col1' in locals() and 'col2' in locals():
    # 評估函數（從RAGEvaluatorTwoModels複製並修改）
    def extract_keywords(text: str) -> List[str]:
        """從應回答詞彙中提取關鍵詞"""
        if pd.isna(text):
            return []
        
        text = re.sub(r'\d+\.', '', text)
        text = re.sub(r'[：:。，,、\(\)]', ' ', text)
        
        keywords = []
        special_terms = [
            "工作許可證", "施工轄區", "包商名稱", "作業內容",
            "承包商現場負責人", "工安業務主管", "施工人員",
            "煙火管制區", "電焊", "切割", "烘烤"
        ]
        
        for term in special_terms:
            if term in text:
                keywords.append(term)
                text = text.replace(term, " ")
        
        jieba.setLogLevel(20)
        words = jieba.cut(text)
        for word in words:
            if len(word.strip()) > 1 and word.strip() not in keywords:
                keywords.append(word.strip())
        
        return keywords
    
    def calculate_coverage_score(answer: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """計算關鍵詞覆蓋率評分"""
        if pd.isna(answer) or not keywords:
            return 0.0, []
        
        matched_keywords = []
        answer_lower = answer.lower()
        
        for keyword in keywords:
            if keyword.lower() in answer_lower:
                matched_keywords.append(keyword)
        
        coverage_rate = len(matched_keywords) / len(keywords) if keywords else 0
        return coverage_rate * 100, matched_keywords
    
    def evaluate_faithfulness(answer: str, reference_keywords: List[str]) -> Tuple[float, str]:
        """評估忠誠度"""
        if pd.isna(answer):
            return 100, "無回答"
        
        numbers_in_answer = re.findall(r'\b\d+\b', answer)
        dates_in_answer = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}', answer)
        
        reference_text = ' '.join(reference_keywords)
        numbers_in_ref = re.findall(r'\b\d+\b', reference_text)
        dates_in_ref = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}', reference_text)
        
        extra_numbers = [n for n in numbers_in_answer if n not in numbers_in_ref]
        extra_dates = [d for d in dates_in_answer if d not in dates_in_ref]
        
        explanation_words = ["因此", "所以", "包括", "例如", "如", "即", "也就是", "用於", "目的"]
        explanation_count = sum(1 for word in explanation_words if word in answer)
        
        if len(extra_numbers) > 2 or len(extra_dates) > 1:
            return 50, "中度忠實"
        elif len(extra_numbers) > 0 or len(extra_dates) > 0:
            return 75, "高度忠實"
        elif explanation_count > 3:
            return 90, "極高忠實"
        else:
            return 100, "完全忠實"
    
    # 執行評估
    if st.button("🚀 開始評估", type="primary"):
        with st.spinner("評估中..."):
            # 準備評估資料
            results = df.copy()
            
            # 計算評分
            for idx, row in results.iterrows():
                keywords = extract_keywords(row[keywords_col])
                
                # 第一個欄位評分
                score1, matched1 = calculate_coverage_score(row[col1], keywords)
                faith1, faith_desc1 = evaluate_faithfulness(row[col1], keywords)
                
                # 第二個欄位評分
                score2, matched2 = calculate_coverage_score(row[col2], keywords)
                faith2, faith_desc2 = evaluate_faithfulness(row[col2], keywords)
                
                # 儲存結果
                results.at[idx, 'SCORE_1'] = score1
                results.at[idx, 'SCORE_2'] = score2
                results.at[idx, 'FAITH_1'] = faith1
                results.at[idx, 'FAITH_2'] = faith2
                results.at[idx, 'MATCHED_1'] = ', '.join(matched1)
                results.at[idx, 'MATCHED_2'] = ', '.join(matched2)
            
            # 計算綜合評分和改善
            results['TOTAL_1'] = results['SCORE_1'] * coverage_weight + results['FAITH_1'] * faithfulness_weight
            results['TOTAL_2'] = results['SCORE_2'] * coverage_weight + results['FAITH_2'] * faithfulness_weight
            results['COVERAGE_IMP'] = results['SCORE_2'] - results['SCORE_1']
            results['FAITH_IMP'] = results['FAITH_2'] - results['FAITH_1']
            results['TOTAL_IMP'] = results['TOTAL_2'] - results['TOTAL_1']
            
            st.session_state['results'] = results
            st.session_state['col1'] = col1
            st.session_state['col2'] = col2
            st.session_state['question_col'] = question_col
            st.success("✅ 評估完成！")
    
    # 顯示結果
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # 建立頁籤
        tab1, tab2, tab3, tab4 = st.tabs(["📊 總覽", "📈 詳細分析", "🎯 互動評分", "💾 下載"])
        
        with tab1:
            st.header("評估總覽")
            
            # 統計卡片
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                avg_score1 = results['SCORE_1'].mean()
                avg_score2 = results['SCORE_2'].mean()
                score_lift = ((avg_score2 - avg_score1) / avg_score1 * 100) if avg_score1 > 0 else 0
                
                st.metric(
                    "覆蓋率變化",
                    f"{avg_score2:.1f}%",
                    f"{score_lift:+.1f}%"
                )
            
            with col_b:
                avg_faith1 = results['FAITH_1'].mean()
                avg_faith2 = results['FAITH_2'].mean()
                faith_lift = ((avg_faith2 - avg_faith1) / avg_faith1 * 100) if avg_faith1 > 0 else 0
                
                st.metric(
                    "忠誠度變化",
                    f"{avg_faith2:.1f}%",
                    f"{faith_lift:+.1f}%"
                )
            
            with col_c:
                improved = (results['TOTAL_IMP'] > 0).sum()
                improved_rate = improved / len(results) * 100
                
                st.metric(
                    "改善比例",
                    f"{improved_rate:.1f}%",
                    f"{improved}/{len(results)} 題"
                )
            
            with col_d:
                avg_total_imp = results['TOTAL_IMP'].mean()
                
                st.metric(
                    "平均改善",
                    f"{avg_total_imp:+.1f}%",
                    "綜合評分"
                )
            
            # 對比圖表
            st.markdown("### 📊 欄位對比")
            
            comparison_data = pd.DataFrame({
                '欄位': [col1, col2],
                '平均覆蓋率': [results['SCORE_1'].mean(), results['SCORE_2'].mean()],
                '平均忠誠度': [results['FAITH_1'].mean(), results['FAITH_2'].mean()],
                '平均綜合評分': [results['TOTAL_1'].mean(), results['TOTAL_2'].mean()]
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='覆蓋率',
                x=comparison_data['欄位'],
                y=comparison_data['平均覆蓋率'],
                text=[f"{v:.1f}%" for v in comparison_data['平均覆蓋率']],
                textposition='auto',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='忠誠度',
                x=comparison_data['欄位'],
                y=comparison_data['平均忠誠度'],
                text=[f"{v:.1f}%" for v in comparison_data['平均忠誠度']],
                textposition='auto',
                marker_color='lightgreen'
            ))
            
            fig.add_trace(go.Bar(
                name='綜合評分',
                x=comparison_data['欄位'],
                y=comparison_data['平均綜合評分'],
                text=[f"{v:.1f}%" for v in comparison_data['平均綜合評分']],
                textposition='auto',
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="各項指標對比",
                yaxis_title="分數 (%)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("詳細分析")
            
            # 改善分布
            fig_hist = px.histogram(
                results,
                x='TOTAL_IMP',
                nbins=20,
                title="總體改善分布",
                labels={'TOTAL_IMP': '改善幅度 (%)'}
            )
            
            avg_imp = results['TOTAL_IMP'].mean()
            fig_hist.add_vline(
                x=avg_imp,
                line_dash="dash",
                line_color="red",
                annotation_text=f"平均: {avg_imp:.1f}%"
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # 問題列表
            st.markdown("### 📋 詳細結果")
            
            display_cols = [
                question_col, 
                'SCORE_1', 'SCORE_2', 'COVERAGE_IMP',
                'FAITH_1', 'FAITH_2', 'FAITH_IMP',
                'TOTAL_1', 'TOTAL_2', 'TOTAL_IMP'
            ]
            
            display_df = results[display_cols].copy()
            display_df.columns = [
                '測試問題',
                f'{col1[:10]}...覆蓋率', f'{col2[:10]}...覆蓋率', '覆蓋率改善',
                f'{col1[:10]}...忠誠度', f'{col2[:10]}...忠誠度', '忠誠度改善',
                f'{col1[:10]}...綜合', f'{col2[:10]}...綜合', '總體改善'
            ]
            
            # 格式化數值
            for col in display_df.columns[1:]:
                display_df[col] = display_df[col].round(1)
            
            st.dataframe(display_df, use_container_width=True, height=500)
        
        with tab3:
            st.header("互動式評分")
            
            # 問題選擇
            idx = st.selectbox(
                "選擇問題",
                range(len(results)),
                format_func=lambda x: f"問題 {x+1}: {results.iloc[x][question_col][:50]}..."
            )
            
            current = results.iloc[idx]
            
            # 顯示問題
            st.info(f"**測試問題**: {current[question_col]}")
            st.write(f"**應回答詞彙**: {current[keywords_col]}")
            
            # 並排顯示
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown(f"### {col1}")
                st.write(f"覆蓋率: {current['SCORE_1']:.1f}%")
                st.write(f"忠誠度: {current['FAITH_1']:.1f}%")
                st.write(f"綜合評分: {current['TOTAL_1']:.1f}%")
                
                with st.expander("查看回答"):
                    st.write(current[col1])
                
                score1 = st.slider(f"您的評分", 1, 5, 3, key=f"s1_{idx}")
            
            with c2:
                st.markdown(f"### {col2}")
                st.write(f"覆蓋率: {current['SCORE_2']:.1f}% ({current['COVERAGE_IMP']:+.1f}%)")
                st.write(f"忠誠度: {current['FAITH_2']:.1f}% ({current['FAITH_IMP']:+.1f}%)")
                st.write(f"綜合評分: {current['TOTAL_2']:.1f}% ({current['TOTAL_IMP']:+.1f}%)")
                
                with st.expander("查看回答"):
                    st.write(current[col2])
                
                score2 = st.slider(f"您的評分", 1, 5, 3, key=f"s2_{idx}")
            
            comment = st.text_area("評語", key=f"comment_{idx}")
            
            if st.button("儲存評分"):
                if f'user_scores' not in st.session_state:
                    st.session_state.user_scores = {}
                
                st.session_state.user_scores[idx] = {
                    'score1': score1,
                    'score2': score2,
                    'comment': comment,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.success("✅ 評分已儲存")
        
        with tab4:
            st.header("下載結果")
            
            if st.button("生成報告", type="primary"):
                # 準備匯出資料
                export_df = results.copy()
                
                # 添加統計
                summary = pd.DataFrame({
                    '統計項目': [
                        f'{col1} 平均覆蓋率',
                        f'{col2} 平均覆蓋率',
                        '覆蓋率平均改善',
                        f'{col1} 平均忠誠度',
                        f'{col2} 平均忠誠度',
                        '忠誠度平均改善',
                        f'{col1} 平均綜合評分',
                        f'{col2} 平均綜合評分',
                        '綜合評分平均改善'
                    ],
                    '數值': [
                        f"{results['SCORE_1'].mean():.2f}%",
                        f"{results['SCORE_2'].mean():.2f}%",
                        f"{results['COVERAGE_IMP'].mean():.2f}%",
                        f"{results['FAITH_1'].mean():.2f}%",
                        f"{results['FAITH_2'].mean():.2f}%",
                        f"{results['FAITH_IMP'].mean():.2f}%",
                        f"{results['TOTAL_1'].mean():.2f}%",
                        f"{results['TOTAL_2'].mean():.2f}%",
                        f"{results['TOTAL_IMP'].mean():.2f}%"
                    ]
                })
                
                # 寫入Excel
                filename = f'比較結果_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                
                with pd.ExcelWriter(filename) as writer:
                    export_df.to_excel(writer, sheet_name='詳細結果', index=False)
                    summary.to_excel(writer, sheet_name='統計摘要', index=False)
                
                # 下載
                with open(filename, 'rb') as f:
                    st.download_button(
                        "📥 下載報告",
                        f,
                        filename,
                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                
                st.success(f"✅ 報告已生成: {filename}")

else:
    st.info("👈 請上傳Excel檔案開始評估")
    
    st.markdown("""
    ### 📖 使用說明
    
    這個系統讓您可以：
    1. **自由選擇**任意兩個欄位進行比較
    2. **彈性設定**測試問題和應回答詞彙欄位
    3. **自訂權重**調整覆蓋率和忠誠度的重要性
    4. **互動評分**提供人工評估和評語
    
    特別適合：
    - 欄位名稱不固定的情況
    - 需要比較多種不同組合
    - 實驗性的比較分析
    """)

# 頁尾
st.markdown("---")
st.caption("RAG 彈性比較系統 v1.0 | 適應各種欄位結構的比較需求")