import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import os
from rag_evaluation_two_models import RAGEvaluatorTwoModels

# 設定頁面配置
st.set_page_config(
    page_title="RAG 原始版本 vs 彙整版本 比較儀表板",
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
st.title("🆚 RAG 原始版本 vs 彙整版本 比較儀表板")
st.markdown("### 透過資料彙整優化提升UPGPT理解能力")

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
        # 本地資料夾路徑
        import os
        # 使用相對路徑或絕對路徑
        try:
            # 先嘗試相對路徑
            data_folder = "test_data"
            if not os.path.exists(data_folder):
                # 如果相對路徑不存在，嘗試從當前目錄
                data_folder = os.path.join(os.getcwd(), "test_data")
        except:
            data_folder = "test_data"
        
        # 確保資料夾存在
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            st.info(f"已建立資料夾：{data_folder}")
        
        # 顯示資料夾路徑以便除錯
        st.caption(f"資料夾路徑：{data_folder}")
        
        # 獲取資料夾中的Excel和CSV檔案
        try:
            all_files = os.listdir(data_folder)
            excel_files = [f for f in all_files 
                          if f.endswith(('.xlsx', '.xls', '.csv')) and not f.startswith('~') and not f.startswith('.')]
            
            # 顯示偵測到的檔案以便除錯
            if all_files:
                st.caption(f"資料夾中的所有檔案：{all_files}")
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
            
            # 顯示檔案資訊
            file_info = os.stat(selected_file_path)
            st.info(f"檔案大小：{file_info.st_size / 1024:.1f} KB")
            st.success(f"✅ 已載入: {selected_file}")
        else:
            st.warning("⚠️ test_data 資料夾中沒有找到 Excel 或 CSV 檔案")
            
            # 顯示資料夾中的檔案（如果有）
            if all_files:
                st.info(f"資料夾中發現的檔案：{', '.join(all_files)}")
            
            st.markdown("""
            請將 Excel (.xlsx, .xls) 或 CSV (.csv) 檔案放入以下路徑：
            ```
            ./test_data/
            ```
            
            **注意事項：**
            - 檔案名稱不能以 `~` 或 `.` 開頭
            - 支援的檔案格式：.xlsx, .xls, .csv
            - 確保檔案有正確的副檔名
            """)
            
            # 顯示一些範例檔案
            st.markdown("**範例檔案名稱：**")
            st.code("""
            ✓ AI指導員_測試腳本_v2拷貝.xlsx
            ✓ 測試結果驗證.csv
            ✓ RAG評估資料.xls
            """)
    
    else:  # 上傳檔案
        uploaded_file = st.file_uploader(
            "上傳測試結果Excel/CSV檔案",
            type=['xlsx', 'xls', 'csv'],
            help="請上傳包含向量知識庫(原始版)和智慧文檔知識庫(彙整版)回答的測試結果"
        )
        
        if uploaded_file is not None:
            # 根據檔案類型保存上傳的檔案
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
            ["向量知識庫", "關鍵字知識庫"],
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
    
    # 評分權重設定
    st.markdown("### ⚖️ 評分權重設定")
    st.info("🔍 調整評估指標在綜合評分中的比重")
    
    coverage_weight = st.slider(
        "覆蓋率權重",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="覆蓋率在綜合評分中的權重"
    )
    faithfulness_weight = 1.0 - coverage_weight
    st.metric("忠誠度權重", f"{faithfulness_weight:.1f}")
    
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
    if isinstance(uploaded_file, str):  # 本地資料夾選擇的檔案
        temp_file_path = uploaded_file  # 直接使用檔案路徑
    else:  # 上傳的檔案
        temp_file_path = "temp_comparison_file.xlsx"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # 根據選擇的知識庫類型建立評估器
    try:
        if original_kb == "向量知識庫" and optimized_kb == "智慧文檔知識庫":
            # 跨技術比較模式
            evaluator = RAGEvaluatorTwoModels(temp_file_path, model_type="cross")
        elif original_kb == "向量知識庫":
            evaluator = RAGEvaluatorTwoModels(temp_file_path, model_type="vector")
        else:
            evaluator = RAGEvaluatorTwoModels(temp_file_path, model_type="smart_doc")
        
        st.session_state.evaluator_instance = evaluator
        
        # 執行評估
        with st.spinner("正在進行深度詞彙分析與評估..."):
            results_df = evaluator.evaluate_all()
            st.session_state.comparison_results = results_df
            
        # 清理臨時檔案
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    except Exception as e:
        st.error(f"❌ 評估過程中發生錯誤：{str(e)}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        st.stop()
    
    # 建立頁籤
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 評估總覽", "📈 互動評分", "📐 改善分析", "💬 問題導覽", "🗂️ 關鍵發現", "📥 下載結果"]
    )
    
    with tab1:
        st.markdown("### 評估總覽")
        
        # 正在評估提示
        st.info(f"正在評估：{original_kb} vs {optimized_kb} | 資料筆數：{len(results_df)} 筆")
        
        # 獲取統計數據
        stats = evaluator.generate_summary_stats()
        
        # 計算關鍵指標
        avg_original_coverage = stats['原始版本']['平均覆蓋率']
        avg_optimized_coverage = stats['彙整優化版本']['平均覆蓋率']
        avg_original_faith = stats['原始版本']['平均忠誠度']
        avg_optimized_faith = stats['彙整優化版本']['平均忠誠度']
        
        coverage_lift = stats['改善效果']['平均覆蓋率提升']
        faith_change = stats['改善效果']['平均忠誠度提升']
        
        significant_improvements = (results_df['TOTAL_IMPROVEMENT'] >= improvement_threshold).sum()
        improvement_rate = stats['改善效果']['顯著改善比例']
        
        # 需要注意的題目（覆蓋率降低或忠誠度大幅下降）
        attention_needed = ((results_df['COVERAGE_IMPROVEMENT'] < 0) | 
                          (results_df['FAITHFULNESS_IMPROVEMENT'] < -20)).sum()
        attention_rate = attention_needed / len(results_df) * 100
        
        # 關鍵指標卡片
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            col1_1, col1_2 = st.columns([5, 1])
            with col1_1:
                st.markdown("**覆蓋率提升**")
            with col1_2:
                st.markdown(
                    "<span title='覆蓋率提升：優化版本相較於原始版本，在回答中包含應回答詞彙比例的改善程度'>ⓘ</span>",
                    unsafe_allow_html=True
                )
            color = '#28a745' if coverage_lift > 0 else '#dc3545'
            st.markdown(f"<h1 style='color: {color}; margin: 0;'>{avg_optimized_coverage:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if coverage_lift > 0 else '↓'} {abs(coverage_lift):.1f}%</p>", unsafe_allow_html=True)
        
        with metric_col2:
            col2_1, col2_2 = st.columns([5, 1])
            with col2_1:
                st.markdown("**忠誠度變化**")
            with col2_2:
                st.markdown(
                    "<span title='忠誠度變化：優化版本相較於原始版本，在AI回答忠實於原始資料程度的變化'>ⓘ</span>",
                    unsafe_allow_html=True
                )
            color = '#28a745' if faith_change >= 0 else '#dc3545'
            st.markdown(f"<h1 style='color: {color}; margin: 0;'>{avg_optimized_faith:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {color}; font-size: 18px;'>{'↑' if faith_change >= 0 else '↓'} {abs(faith_change):.1f}%</p>", unsafe_allow_html=True)
        
        with metric_col3:
            col3_1, col3_2 = st.columns([5, 1])
            with col3_1:
                st.markdown("**顯著改善率**")
            with col3_2:
                st.markdown(
                    "<span title='顯著改善率：綜合評分提升超過10%的問題佔總問題數的比例'>ⓘ</span>",
                    unsafe_allow_html=True
                )
            st.markdown(f"<h1 style='color: #28a745; margin: 0;'>{improvement_rate:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #28a745; font-size: 18px;'>↑ {significant_improvements} 題</p>", unsafe_allow_html=True)
        
        with metric_col4:
            col4_1, col4_2 = st.columns([5, 1])
            with col4_1:
                st.markdown("**需要注意比例**")
            with col4_2:
                st.markdown(
                    "<span title='需要注意比例：覆蓋率降低或忠誠度大幅下降（>20%）的問題佔總問題數的比例'>ⓘ</span>",
                    unsafe_allow_html=True
                )
            color = '#ffc107' if attention_rate > 20 else '#28a745'
            st.markdown(f"<h1 style='color: {color}; margin: 0;'>{attention_rate:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: {color}; font-size: 18px;'>↑ {attention_needed} 題</p>", unsafe_allow_html=True)
        
        # 詳細指標對比
        st.markdown("### 📊 詳細指標對比")
        
        # 建立更詳細的對比數據
        comparison_metrics = [
            {
                '評估版本': '🔴 原始版本',
                '平均覆蓋率': avg_original_coverage,
                '平均忠誠度': avg_original_faith,
                '平均綜合評分': stats['原始版本']['平均綜合評分'],
                '高覆蓋率比例': stats['原始版本']['高覆蓋率比例'],
                '完全忠實比例': stats['原始版本']['完全忠實比例']
            },
            {
                '評估版本': '🟢 彙整優化版本',
                '平均覆蓋率': avg_optimized_coverage,
                '平均忠誠度': avg_optimized_faith,
                '平均綜合評分': stats['彙整優化版本']['平均綜合評分'],
                '高覆蓋率比例': stats['彙整優化版本']['高覆蓋率比例'],
                '完全忠實比例': stats['彙整優化版本']['完全忠實比例']
            }
        ]
        
        # 添加改善幅度行
        improvement_row = {
            '評估版本': '📊 改善幅度',
            '平均覆蓋率': coverage_lift,
            '平均忠誠度': faith_change,
            '平均綜合評分': stats['改善效果']['平均綜合評分提升'],
            '高覆蓋率比例': stats['彙整優化版本']['高覆蓋率比例'] - stats['原始版本']['高覆蓋率比例'],
            '完全忠實比例': stats['彙整優化版本']['完全忠實比例'] - stats['原始版本']['完全忠實比例']
        }
        
        # 創建DataFrame
        comparison_df = pd.DataFrame(comparison_metrics)
        improvement_df = pd.DataFrame([improvement_row])
        
        # 定義格式化函數
        def format_cell_value(val, col_name, row_idx):
            if row_idx < 2:  # 原始和優化版本
                return f"{val:.1f}%"
            else:  # 改善幅度行
                color = '#2ecc71' if val > 0 else '#e74c3c' if val < 0 else '#95a5a6'
                arrow = '↑' if val > 0 else '↓' if val < 0 else '='
                return f"<span style='color: {color}; font-weight: bold;'>{arrow} {abs(val):.1f}%</span>"
        
        # 創建HTML表格
        html_table = "<table style='width: 100%; border-collapse: collapse; background-color: #1a1a1a;'>"
        html_table += "<thead><tr style='background-color: #2d2d2d;'>"
        
        # 表頭
        columns = ['評估版本', '平均覆蓋率', '平均忠誠度', '平均綜合評分', '高覆蓋率比例', '完全忠實比例']
        for col in columns:
            html_table += f"<th style='padding: 12px; border-bottom: 2px solid #444; text-align: left; color: #ffffff;'>{col}</th>"
        html_table += "</tr></thead><tbody>"
        
        # 數據行
        all_data = pd.concat([comparison_df, improvement_df], ignore_index=True)
        
        for idx, row in all_data.iterrows():
            if idx == 2:  # 改善幅度行
                bg_color = '#2a2a2a'
                border_top = 'border-top: 2px solid #444;'
            else:
                bg_color = '#1a1a1a'
                border_top = ''
            
            html_table += f"<tr style='background-color: {bg_color};'>"
            
            for col_idx, col in enumerate(columns):
                cell_value = row[col]
                style = f"padding: 12px; border-bottom: 1px solid #333; color: #ffffff; {border_top if idx == 2 else ''}"
                
                if col_idx == 0:  # 第一列（評估版本）
                    html_table += f"<td style='{style} font-weight: bold;'>{cell_value}</td>"
                else:
                    if idx < 2:  # 原始和優化版本
                        # 根據數值大小添加背景色
                        if col in ['平均覆蓋率', '平均綜合評分', '高覆蓋率比例']:
                            # 這些指標越高越好
                            if cell_value >= 80:
                                bg = 'background: linear-gradient(90deg, rgba(46, 204, 113, 0.3) 0%, rgba(46, 204, 113, 0.15) 100%);'
                            elif cell_value >= 60:
                                bg = 'background: linear-gradient(90deg, rgba(243, 156, 18, 0.3) 0%, rgba(243, 156, 18, 0.15) 80%);'
                            else:
                                bg = 'background: linear-gradient(90deg, rgba(231, 76, 60, 0.3) 0%, rgba(231, 76, 60, 0.15) 60%);'
                        elif col == '平均忠誠度':
                            # 忠誠度接近100最好
                            if cell_value >= 90:
                                bg = 'background: linear-gradient(90deg, rgba(46, 204, 113, 0.3) 0%, rgba(46, 204, 113, 0.15) 100%);'
                            elif cell_value >= 70:
                                bg = 'background: linear-gradient(90deg, rgba(243, 156, 18, 0.3) 0%, rgba(243, 156, 18, 0.15) 80%);'
                            else:
                                bg = 'background: linear-gradient(90deg, rgba(231, 76, 60, 0.3) 0%, rgba(231, 76, 60, 0.15) 60%);'
                        elif col == '完全忠實比例':
                            # 完全忠實比例也是越高越好
                            if cell_value >= 80:
                                bg = 'background: linear-gradient(90deg, rgba(46, 204, 113, 0.3) 0%, rgba(46, 204, 113, 0.15) 100%);'
                            elif cell_value >= 60:
                                bg = 'background: linear-gradient(90deg, rgba(243, 156, 18, 0.3) 0%, rgba(243, 156, 18, 0.15) 80%);'
                            else:
                                bg = 'background: linear-gradient(90deg, rgba(231, 76, 60, 0.3) 0%, rgba(231, 76, 60, 0.15) 60%);'
                        else:
                            bg = ''
                        
                        html_table += f"<td style='{style} {bg}'>{cell_value:.1f}%</td>"
                    else:  # 改善幅度行
                        formatted_val = format_cell_value(cell_value, col, idx)
                        html_table += f"<td style='{style}'>{formatted_val}</td>"
            
            html_table += "</tr>"
        
        html_table += "</tbody></table>"
        
        # 顯示表格
        st.markdown(html_table, unsafe_allow_html=True)
        
        # 添加指標說明
        with st.expander("📖 指標說明"):
            st.markdown("""
            - **平均覆蓋率**: 回答中包含應回答詞彙的比例（越高越好）
            - **平均忠誠度**: AI回答忠實於原始資料的程度（越高越好）
            - **平均綜合評分**: 覆蓋率 × 0.5 + 忠誠度 × 0.5（越高越好）
            - **高覆蓋率比例**: 覆蓋率≥80%的問題佔比（越高越好）
            - **完全忠實比例**: 完全不虛構內容的問題佔比（越高越好）
            
            🟢 綠色背景：表現優秀（≥80分）
            🟡 黃色背景：表現良好（60-79分）
            🔴 紅色背景：需要改善（<60分）
            """)
        
        # 分隔線
        st.markdown("---")
        
        # 覆蓋率對比圖表
        st.markdown("### 📊 覆蓋率對比")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # 覆蓋率對比柱狀圖
            coverage_comp = pd.DataFrame({
                '版本': ['原始版本', '彙整優化版本'],
                '覆蓋率 (%)': [avg_original_coverage, avg_optimized_coverage]
            })
            
            fig_coverage_bar = px.bar(
                coverage_comp, 
                x='版本', 
                y='覆蓋率 (%)',
                text='覆蓋率 (%)',
                color='版本',
                color_discrete_map={'原始版本': '#e57373', '彙整優化版本': '#81c784'},
                title='覆蓋率對比'
            )
            
            fig_coverage_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_coverage_bar.update_layout(showlegend=False, height=300)
            fig_coverage_bar.update_yaxes(range=[0, 100])
            
            st.plotly_chart(fig_coverage_bar, use_container_width=True)
        
        with col_chart2:
            # 忠誠度對比柱狀圖
            faith_comp = pd.DataFrame({
                '版本': ['原始版本', '彙整優化版本'],
                '忠誠度 (%)': [avg_original_faith, avg_optimized_faith]
            })
            
            fig_faith_bar = px.bar(
                faith_comp, 
                x='版本', 
                y='忠誠度 (%)',
                text='忠誠度 (%)',
                color='版本',
                color_discrete_map={'原始版本': '#e57373', '彙整優化版本': '#81c784'},
                title='忠誠度對比'
            )
            
            fig_faith_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_faith_bar.update_layout(showlegend=False, height=300)
            fig_faith_bar.update_yaxes(range=[0, 100])
            
            st.plotly_chart(fig_faith_bar, use_container_width=True)
    
    with tab2:
        st.markdown("### 📈 互動評分")
        st.info("透過人工評分深入了解AI回答品質的實際改善")
        
        # 問題選擇
        question_idx = st.selectbox(
            "選擇要評分的問題",
            range(len(results_df)),
            format_func=lambda x: f"問題 {x+1}: {results_df.iloc[x]['測試問題'][:50]}..."
        )
        
        current_result = results_df.iloc[question_idx]
        
        # 顯示問題和應回答詞彙
        st.markdown("#### 📝 測試問題")
        st.info(current_result['測試問題'])
        
        st.markdown("#### 🎯 應回答詞彙")
        keywords_text = current_result.get('應回答之詞彙', '')
        st.success(keywords_text)
        
        # 顯示關鍵詞分析
        keywords = evaluator.extract_keywords(keywords_text)
        st.markdown(f"**關鍵詞總數**: {len(keywords)} 個")
        with st.expander("查看關鍵詞列表"):
            st.write(", ".join(keywords))
        
        # 並排顯示兩個版本
        col_original, col_optimized = st.columns(2)
        
        with col_original:
            st.markdown("#### 🔴 原始版本（向量知識庫）")
            
            # AI評分
            st.metric("覆蓋率", f"{current_result['SCORE_ORIGINAL']:.1f}%")
            st.metric("忠誠度", f"{current_result['FAITHFULNESS_ORIGINAL']:.0f}%")
            
            # 匹配的關鍵詞
            matched_keywords_orig = current_result['MATCHED_KEYWORDS_ORIGINAL'].split(', ') if current_result['MATCHED_KEYWORDS_ORIGINAL'] else []
            with st.expander(f"匹配關鍵詞 ({len(matched_keywords_orig)}/{len(keywords)})"):
                if matched_keywords_orig and matched_keywords_orig != ['']:
                    st.success(", ".join(matched_keywords_orig))
                else:
                    st.warning("無匹配關鍵詞")
            
            # 回答內容
            st.markdown("**回答內容**")
            st.text_area("", value=current_result['ANSWER_ORIGINAL'], height=200, key=f"orig_{question_idx}")
            
            # 忠誠度分析
            st.markdown(f"**忠誠度類型**: {current_result['FAITHFULNESS_DESC_ORIGINAL']}")
        
        with col_optimized:
            st.markdown("#### 🟢 優化版本（智慧文檔知識庫）")
            
            # AI評分和改善
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "覆蓋率", 
                    f"{current_result['SCORE_OPTIMIZED']:.1f}%",
                    f"{current_result['COVERAGE_IMPROVEMENT']:.1f}%"
                )
            with col2:
                st.metric(
                    "忠誠度", 
                    f"{current_result['FAITHFULNESS_OPTIMIZED']:.0f}%",
                    f"{current_result['FAITHFULNESS_IMPROVEMENT']:.0f}%"
                )
            
            # 匹配的關鍵詞
            matched_keywords_opt = current_result['MATCHED_KEYWORDS_OPTIMIZED'].split(', ') if current_result['MATCHED_KEYWORDS_OPTIMIZED'] else []
            with st.expander(f"匹配關鍵詞 ({len(matched_keywords_opt)}/{len(keywords)})"):
                if matched_keywords_opt and matched_keywords_opt != ['']:
                    st.success(", ".join(matched_keywords_opt))
                else:
                    st.warning("無匹配關鍵詞")
            
            # 新增匹配的關鍵詞
            if len(matched_keywords_opt) > len(matched_keywords_orig):
                new_keywords = [k for k in matched_keywords_opt if k not in matched_keywords_orig]
                if new_keywords:
                    with st.expander(f"✨ 新增匹配 ({len(new_keywords)})"):
                        st.success(", ".join(new_keywords))
            
            # 回答內容
            st.markdown("**回答內容**")
            st.text_area("", value=current_result['ANSWER_OPTIMIZED'], height=200, key=f"opt_{question_idx}")
            
            # 忠誠度分析
            st.markdown(f"**忠誠度類型**: {current_result['FAITHFULNESS_DESC_OPTIMIZED']}")
    
    with tab3:
        st.markdown("### 📐 改善分析")
        st.info("分析各題目的改善情況，識別優化策略的效果模式")
        
        # 改善分布分析
        col1, col2 = st.columns(2)
        
        with col1:
            # 覆蓋率改善分布
            fig_coverage_dist = px.histogram(
                results_df,
                x='COVERAGE_IMPROVEMENT',
                nbins=20,
                title='覆蓋率改善分布',
                labels={'COVERAGE_IMPROVEMENT': '改善幅度 (%)'},
                color_discrete_sequence=['#2196F3']
            )
            fig_coverage_dist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_coverage_dist.add_vline(
                x=results_df['COVERAGE_IMPROVEMENT'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"平均: {results_df['COVERAGE_IMPROVEMENT'].mean():.1f}%"
            )
            st.plotly_chart(fig_coverage_dist, use_container_width=True)
        
        with col2:
            # 忠誠度變化分布
            fig_faith_dist = px.histogram(
                results_df,
                x='FAITHFULNESS_IMPROVEMENT',
                nbins=20,
                title='忠誠度變化分布',
                labels={'FAITHFULNESS_IMPROVEMENT': '變化幅度 (%)'},
                color_discrete_sequence=['#4CAF50']
            )
            fig_faith_dist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_faith_dist.add_vline(
                x=results_df['FAITHFULNESS_IMPROVEMENT'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"平均: {results_df['FAITHFULNESS_IMPROVEMENT'].mean():.1f}%"
            )
            st.plotly_chart(fig_faith_dist, use_container_width=True)
        
        # 改善相關性分析
        st.markdown("### 🔍 改善相關性分析")
        
        fig_scatter = px.scatter(
            results_df,
            x='COVERAGE_IMPROVEMENT',
            y='FAITHFULNESS_IMPROVEMENT',
            hover_data=['序號', '測試問題'],
            title="覆蓋率改善 vs 忠誠度變化",
            labels={
                'COVERAGE_IMPROVEMENT': '覆蓋率改善 (%)',
                'FAITHFULNESS_IMPROVEMENT': '忠誠度變化 (%)'
            },
            color_continuous_scale='RdYlGn',
            color='TOTAL_IMPROVEMENT'
        )
        
        # 添加象限分隔線
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
        
        # 添加象限標籤
        fig_scatter.add_annotation(x=20, y=10, text="雙重改善", showarrow=False, font=dict(size=12, color="green"))
        fig_scatter.add_annotation(x=-20, y=10, text="忠誠度改善", showarrow=False, font=dict(size=12, color="blue"))
        fig_scatter.add_annotation(x=20, y=-10, text="覆蓋率改善", showarrow=False, font=dict(size=12, color="orange"))
        fig_scatter.add_annotation(x=-20, y=-10, text="雙重退步", showarrow=False, font=dict(size=12, color="red"))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        st.markdown("### 💬 問題導覽")
        st.info("快速瀏覽所有測試問題及其回答比較")
        
        # 篩選選項
        filter_option = st.selectbox(
            "篩選顯示",
            ["所有問題", "顯著改善", "略有改善", "無變化", "效果退步"]
        )
        
        # 根據條件篩選
        if filter_option == "顯著改善":
            filtered_df = results_df[results_df['TOTAL_IMPROVEMENT'] >= improvement_threshold]
        elif filter_option == "略有改善":
            filtered_df = results_df[(results_df['TOTAL_IMPROVEMENT'] > 0) & (results_df['TOTAL_IMPROVEMENT'] < improvement_threshold)]
        elif filter_option == "無變化":
            filtered_df = results_df[results_df['TOTAL_IMPROVEMENT'] == 0]
        elif filter_option == "效果退步":
            filtered_df = results_df[results_df['TOTAL_IMPROVEMENT'] < 0]
        else:
            filtered_df = results_df
        
        # 顯示問題列表
        for idx, row in filtered_df.iterrows():
            with st.expander(f"問題 {row['序號']}: {row['測試問題'][:50]}..."):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    keywords_text = row.get('應回答之詞彙', '')
                    st.markdown(f"**應回答詞彙**: {keywords_text[:100]}...")
                
                with col2:
                    st.metric("覆蓋率改善", f"{row['COVERAGE_IMPROVEMENT']:.1f}%")
                
                with col3:
                    st.metric("忠誠度變化", f"{row['FAITHFULNESS_IMPROVEMENT']:.0f}%")
                
                # 顯示關鍵詞匹配情況
                st.markdown("**關鍵詞匹配分析**")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("原始版本")
                    matched_orig = len(row['MATCHED_KEYWORDS_ORIGINAL'].split(', ')) if row['MATCHED_KEYWORDS_ORIGINAL'] else 0
                    keywords = evaluator.extract_keywords(keywords_text)
                    st.write(f"匹配: {matched_orig}/{len(keywords)}")
                    st.write(f"覆蓋率: {row['SCORE_ORIGINAL']:.1f}%")
                
                with col_b:
                    st.markdown("優化版本")
                    matched_opt = len(row['MATCHED_KEYWORDS_OPTIMIZED'].split(', ')) if row['MATCHED_KEYWORDS_OPTIMIZED'] else 0
                    st.write(f"匹配: {matched_opt}/{len(keywords)}")
                    st.write(f"覆蓋率: {row['SCORE_OPTIMIZED']:.1f}%")
                    if matched_opt > matched_orig:
                        st.success(f"新增 {matched_opt - matched_orig} 個關鍵詞")
    
    with tab5:
        st.markdown("### 🗂️ 關鍵發現")
        st.info("基於評估結果的重要洞察與建議")
        
        # 分析關鍵發現
        findings_col1, findings_col2 = st.columns(2)
        
        with findings_col1:
            st.markdown("#### 📊 數據洞察")
            
            improved_count = (results_df['COVERAGE_IMPROVEMENT'] > 0).sum()
            declined_count = (results_df['COVERAGE_IMPROVEMENT'] < 0).sum()
            unchanged_count = (results_df['COVERAGE_IMPROVEMENT'] == 0).sum()
            
            st.markdown(f"""
            - ✅ **改善題數**: {improved_count} 題 ({improved_count/len(results_df)*100:.1f}%)
            - ❌ **退步題數**: {declined_count} 題 ({declined_count/len(results_df)*100:.1f}%)
            - ➖ **無變化題數**: {unchanged_count} 題 ({unchanged_count/len(results_df)*100:.1f}%)
            - 📈 **平均改善幅度**: {results_df['COVERAGE_IMPROVEMENT'].mean():.1f}%
            - 📊 **改善中位數**: {results_df['COVERAGE_IMPROVEMENT'].median():.1f}%
            """)
            
            # 關鍵詞分析
            total_keywords = sum(len(evaluator.extract_keywords(row['應回答之詞彙'])) for _, row in results_df.iterrows())
            avg_keywords_per_question = total_keywords / len(results_df)
            
            st.markdown(f"""
            #### 🔤 關鍵詞分析
            - 📝 **總關鍵詞數**: {total_keywords} 個
            - 📊 **平均每題關鍵詞**: {avg_keywords_per_question:.1f} 個
            - 🎯 **原始版本平均匹配率**: {results_df['SCORE_ORIGINAL'].mean():.1f}%
            - 🎯 **優化版本平均匹配率**: {results_df['SCORE_OPTIMIZED'].mean():.1f}%
            """)
        
        with findings_col2:
            st.markdown("#### 💡 優化建議")
            
            # 根據數據提供建議
            if results_df['COVERAGE_IMPROVEMENT'].mean() > 10:
                st.success("🎯 智慧文檔知識庫的彙整策略非常有效")
            elif results_df['COVERAGE_IMPROVEMENT'].mean() > 5:
                st.info("📈 優化方向正確，但仍有改善空間")
            else:
                st.warning("⚠️ 優化效果有限，需要調整策略")
            
            # 忠誠度建議
            if results_df['FAITHFULNESS_IMPROVEMENT'].mean() < -10:
                st.warning("⚠️ 注意：優化版本的忠誠度有所下降，可能包含過多推測內容")
            else:
                st.success("✅ 忠誠度保持良好，未出現明顯虛構問題")
            
            # 找出最需要改善的問題類型
            worst_questions = results_df.nsmallest(5, 'COVERAGE_IMPROVEMENT')
            if not worst_questions.empty:
                st.markdown("#### 🔍 需重點關注的問題")
                for _, row in worst_questions.iterrows():
                    st.markdown(f"- 問題 {row['序號']}: 覆蓋率 {row['COVERAGE_IMPROVEMENT']:.1f}%")
        
        # 詞彙深度分析
        st.markdown("### 🔤 詞彙深度分析")
        
        # 收集所有關鍵詞和匹配情況
        all_keywords_original = []
        all_keywords_optimized = []
        
        for _, row in results_df.iterrows():
            # 獲取原始關鍵詞
            keywords = evaluator.extract_keywords(row['應回答之詞彙'])
            
            # 獲取匹配的關鍵詞
            matched_orig = row['MATCHED_KEYWORDS_ORIGINAL'].split(', ') if row['MATCHED_KEYWORDS_ORIGINAL'] else []
            matched_opt = row['MATCHED_KEYWORDS_OPTIMIZED'].split(', ') if row['MATCHED_KEYWORDS_OPTIMIZED'] else []
            
            all_keywords_original.extend(matched_orig)
            all_keywords_optimized.extend(matched_opt)
        
        from collections import Counter
        
        # 分析哪些關鍵詞在優化版本中新增匹配
        original_set = set(all_keywords_original)
        optimized_set = set(all_keywords_optimized)
        new_matched = list(optimized_set - original_set)
        
        if new_matched:
            st.markdown("#### ✅ 優化版本新增匹配的關鍵詞")
            new_matched_df = pd.DataFrame(
                [(kw, all_keywords_optimized.count(kw)) for kw in new_matched[:10]],
                columns=['關鍵詞', '出現次數']
            )
            st.dataframe(new_matched_df, use_container_width=True, hide_index=True)
        
        # 顯示最常匹配的關鍵詞
        keyword_counter = Counter(all_keywords_optimized)
        if keyword_counter:
            st.markdown("#### 🎯 最常匹配的關鍵詞 (優化版本)")
            top_keywords_df = pd.DataFrame(
                keyword_counter.most_common(10),
                columns=['關鍵詞', '出現次數']
            )
            st.dataframe(top_keywords_df, use_container_width=True, hide_index=True)
    
    with tab6:
        st.markdown("### 📥 下載結果")
        st.info("匯出完整評估報告與分析數據")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 評估報告")
            
            if st.button("生成評估報告", type="primary"):
                # 準備報告數據
                report_data = {
                    '測試問題': results_df['測試問題'],
                    '應回答之詞彙': results_df['應回答之詞彙'],
                    '原始覆蓋率': results_df['SCORE_ORIGINAL'],
                    '優化覆蓋率': results_df['SCORE_OPTIMIZED'],
                    '覆蓋率改善': results_df['COVERAGE_IMPROVEMENT'],
                    '原始忠誠度': results_df['FAITHFULNESS_ORIGINAL'],
                    '優化忠誠度': results_df['FAITHFULNESS_OPTIMIZED'],
                    '忠誠度變化': results_df['FAITHFULNESS_IMPROVEMENT'],
                    '原始綜合評分': results_df['TOTAL_SCORE_ORIGINAL'],
                    '優化綜合評分': results_df['TOTAL_SCORE_OPTIMIZED'],
                    '綜合改善': results_df['TOTAL_IMPROVEMENT']
                }
                
                report_df = pd.DataFrame(report_data)
                
                # 生成Excel檔案
                filename = f'RAG比較評估_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # 詳細結果
                    report_df.to_excel(writer, sheet_name='詳細結果', index=False)
                    
                    # 統計摘要
                    summary_data = {
                        '指標': ['平均覆蓋率(原始)', '平均覆蓋率(優化)', '覆蓋率提升',
                                '平均忠誠度(原始)', '平均忠誠度(優化)', '忠誠度變化',
                                '顯著改善題數', '改善比例'],
                        '數值': [
                            f"{results_df['SCORE_ORIGINAL'].mean():.2f}%",
                            f"{results_df['SCORE_OPTIMIZED'].mean():.2f}%",
                            f"{results_df['COVERAGE_IMPROVEMENT'].mean():.2f}%",
                            f"{results_df['FAITHFULNESS_ORIGINAL'].mean():.2f}%",
                            f"{results_df['FAITHFULNESS_OPTIMIZED'].mean():.2f}%",
                            f"{results_df['FAITHFULNESS_IMPROVEMENT'].mean():.2f}%",
                            (results_df['TOTAL_IMPROVEMENT'] > 10).sum(),
                            f"{(results_df['TOTAL_IMPROVEMENT'] > 10).sum() / len(results_df) * 100:.2f}%"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='統計摘要', index=False)
                
                # 提供下載
                with open(filename, 'rb') as f:
                    st.download_button(
                        label="📥 下載評估報告",
                        data=f,
                        file_name=filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                
                # 清理臨時檔案
                if os.path.exists(filename):
                    os.remove(filename)
                
                st.success("✅ 評估報告已生成")
        
        with col2:
            st.markdown("#### 📈 視覺化圖表")
            
            if st.button("生成圖表集", type="secondary"):
                # 創建圖表集
                fig_collection = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('覆蓋率對比', '忠誠度對比', '改善分布', '相關性分析'),
                    specs=[[{"type": "bar"}, {"type": "bar"}],
                          [{"type": "histogram"}, {"type": "scatter"}]]
                )
                
                # 添加覆蓋率對比
                fig_collection.add_trace(
                    go.Bar(x=['原始版本', '優化版本'], 
                          y=[results_df['SCORE_ORIGINAL'].mean(), results_df['SCORE_OPTIMIZED'].mean()],
                          marker_color=['#e57373', '#81c784']),
                    row=1, col=1
                )
                
                # 添加忠誠度對比
                fig_collection.add_trace(
                    go.Bar(x=['原始版本', '優化版本'], 
                          y=[results_df['FAITHFULNESS_ORIGINAL'].mean(), results_df['FAITHFULNESS_OPTIMIZED'].mean()],
                          marker_color=['#e57373', '#81c784']),
                    row=1, col=2
                )
                
                # 添加改善分布
                fig_collection.add_trace(
                    go.Histogram(x=results_df['COVERAGE_IMPROVEMENT'], nbinsx=20),
                    row=2, col=1
                )
                
                # 添加相關性分析
                fig_collection.add_trace(
                    go.Scatter(x=results_df['COVERAGE_IMPROVEMENT'], 
                             y=results_df['FAITHFULNESS_IMPROVEMENT'],
                             mode='markers'),
                    row=2, col=2
                )
                
                fig_collection.update_layout(height=800, showlegend=False)
                
                # 保存圖表
                chart_filename = f'RAG評估圖表_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
                fig_collection.write_html(chart_filename)
                
                with open(chart_filename, 'rb') as f:
                    st.download_button(
                        label="📥 下載圖表",
                        data=f,
                        file_name=chart_filename,
                        mime='text/html'
                    )
                
                # 清理臨時檔案
                if os.path.exists(chart_filename):
                    os.remove(chart_filename)
                
                st.success("✅ 圖表已生成")

else:
    # 未上傳檔案時的提示
    st.info("👈 請從側邊欄上傳測試結果Excel檔案開始評估")
    
    # 使用說明
    with st.expander("📖 使用說明", expanded=True):
        st.markdown("""
        ### 🎯 系統目的
        本系統專門用於比較RAG系統的向量知識庫（原始版本）與智慧文檔知識庫（彙整版本），
        通過深度詞彙分析展示資料彙整策略帶來的理解能力提升。
        
        ### 📊 核心評估指標
        
        1. **覆蓋率** - 衡量AI回答包含多少應回答的關鍵詞
        2. **忠誠度** - 評估AI回答是否忠實於原始資料
        3. **關鍵詞匹配** - 詳細分析每個關鍵詞的匹配情況
        4. **改善分析** - 識別優化策略的效果模式
        
        ### 🚀 開始使用
        
        1. 上傳包含測試結果的Excel檔案（需包含測試問題、應回答詞彙、兩個版本的回答）
        2. 系統將自動進行詞彙深度分析
        3. 查看各項評估指標和改善情況
        4. 導出完整評估報告
        
        ### 📝 Excel檔案格式要求
        
        - **測試問題**: 測試的問題內容
        - **應回答詞彙**: 期望回答包含的關鍵詞彙
        - **ANSWER_1** 或類似: 原始版本（向量知識庫）的回答
        - **ANSWER_2** 或類似: 優化版本（智慧文檔知識庫）的回答
        """)

# 頁尾
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>RAG 原始版本 vs 彙整版本 比較儀表板 v2.0</p>
    <p>透過資料彙整優化提升AI理解能力 | © 2024</p>
</div>
""", unsafe_allow_html=True)