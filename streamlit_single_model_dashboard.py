import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from rag_evaluation_single_model import RAGEvaluatorSingleModel
import os

st.set_page_config(
    page_title="單一模型RAG評估儀表板",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 標題和說明
st.title("🎯 單一模型 RAG 評估儀表板")
st.markdown("### 職災保護QA測試結果分析")

# 側邊欄
with st.sidebar:
    st.header("📁 檔案設定")
    
    # 上傳檔案
    uploaded_file = st.file_uploader(
        "選擇測試結果CSV檔案",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # 保存上傳的檔案
        temp_file_path = "temp_single_model_test.csv"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 執行評估按鈕
        if st.button("🚀 執行評估", type="primary", use_container_width=True):
            with st.spinner("評估中..."):
                evaluator = RAGEvaluatorSingleModel(temp_file_path)
                results = evaluator.evaluate_all()
                summary = evaluator.generate_summary()
                
                # 保存結果到session state
                st.session_state['single_results'] = results
                st.session_state['single_summary'] = summary
                st.session_state['single_evaluator'] = evaluator
            
            st.success("✅ 評估完成！")

# 主要內容區
if 'single_results' in st.session_state:
    results = st.session_state['single_results']
    summary = st.session_state['single_summary']
    evaluator = st.session_state['single_evaluator']
    
    # 建立頁籤
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 總覽", "📈 詳細評分", "🔍 問題分析", "💾 下載結果"]
    )
    
    with tab1:
        st.header("評估總覽")
        
        # 顯示關鍵指標
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📝 總題數",
                value=summary['總題數']
            )
        
        with col2:
            st.metric(
                label="🎯 平均覆蓋率",
                value=f"{summary['平均覆蓋率']:.1f}%",
                delta=f"{summary['高覆蓋率題數']} 題 ≥80%"
            )
        
        with col3:
            st.metric(
                label="🎭 平均忠誠度",
                value=f"{summary['平均忠誠度']:.1f}%",
                delta=f"{summary['高忠誠度題數']} 題 ≥90%"
            )
        
        with col4:
            st.metric(
                label="⭐ 平均綜合評分",
                value=f"{summary['平均綜合評分']:.1f}%",
                delta=f"{summary['優秀綜合評分題數']} 題 ≥85%"
            )
        
        # 評分分布圖
        st.subheader("📊 評分分布")
        
        # 建立子圖
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("覆蓋率分布", "忠誠度分布", "綜合評分分布")
        )
        
        # 覆蓋率分布
        fig.add_trace(
            go.Histogram(
                x=results['覆蓋率分數'],
                nbinsx=20,
                name='覆蓋率',
                marker_color='#3498db'
            ),
            row=1, col=1
        )
        
        # 忠誠度分布
        fig.add_trace(
            go.Histogram(
                x=results['忠誠度分數'],
                nbinsx=20,
                name='忠誠度',
                marker_color='#2ecc71'
            ),
            row=1, col=2
        )
        
        # 綜合評分分布
        fig.add_trace(
            go.Histogram(
                x=results['綜合評分'],
                nbinsx=20,
                name='綜合評分',
                marker_color='#9b59b6'
            ),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="評分分布圖"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("詳細評分結果")
        
        # 顯示詳細資料表
        display_columns = ['編號', '問題', '覆蓋率分數', '忠誠度分數', '綜合評分', '忠誠度描述']
        display_df = results[display_columns].copy()
        
        # 添加條件格式
        def highlight_scores(val):
            if isinstance(val, (int, float)):
                if val >= 85:
                    return 'background-color: #2ecc71'
                elif val >= 70:
                    return 'background-color: #f39c12'
                else:
                    return 'background-color: #e74c3c'
            return ''
        
        # 套用樣式
        styled_df = display_df.style.applymap(
            highlight_scores, 
            subset=['覆蓋率分數', '忠誠度分數', '綜合評分']
        )
        
        st.dataframe(styled_df, use_container_width=True, height=600)
    
    with tab3:
        st.header("問題分析")
        
        # 選擇問題進行深入分析
        question_list = results['問題'].tolist()
        selected_question = st.selectbox(
            "選擇問題進行分析",
            question_list,
            index=0
        )
        
        # 找到選中問題的資料
        selected_data = results[results['問題'] == selected_question].iloc[0]
        
        # 顯示分析結果
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 回答重點（關鍵詞）")
            st.info(selected_data['回答重點'])
            
            st.subheader("📝 提取的關鍵詞")
            keywords = selected_data['關鍵詞列表'].split(', ')
            for keyword in keywords:
                st.write(f"• {keyword}")
        
        with col2:
            st.subheader("🤖 AI回答")
            st.info(selected_data['UPGPT回答'])
            
            st.subheader("✅ 匹配的關鍵詞")
            matched = selected_data['匹配關鍵詞'].split(', ') if selected_data['匹配關鍵詞'] else []
            for keyword in matched:
                st.write(f"• {keyword}")
        
        # 評分詳情
        st.subheader("📊 評分詳情")
        score_col1, score_col2, score_col3 = st.columns(3)
        
        with score_col1:
            st.metric("覆蓋率分數", f"{selected_data['覆蓋率分數']:.1f}%")
        
        with score_col2:
            st.metric("忠誠度分數", f"{selected_data['忠誠度分數']:.1f}%")
        
        with score_col3:
            st.metric("綜合評分", f"{selected_data['綜合評分']:.1f}%")
    
    with tab4:
        st.header("下載評估結果")
        
        # 儲存按鈕
        if st.button("💾 儲存評估結果為Excel", type="primary"):
            output_path = evaluator.save_results("single_model_evaluation")
            st.success(f"✅ 檔案已儲存至: {output_path}")
            
            # 提供下載連結
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="⬇️ 下載Excel檔案",
                    data=f,
                    file_name=os.path.basename(output_path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

else:
    st.info("👈 請從側邊欄上傳CSV檔案並執行評估")