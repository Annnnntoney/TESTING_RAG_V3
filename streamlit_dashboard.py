import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from rag_evaluation_v2 import RAGEvaluatorV2 as RAGEvaluator
import os

st.set_page_config(
    page_title="RAG評估儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 標題和說明
st.title("🤖 RAG LLM 評估儀表板")
st.markdown("### 測試結果精確度分析")

# 側邊欄
with st.sidebar:
    st.header("📁 檔案選擇")
    
    # 檔案選擇方式
    file_source = st.radio(
        "選擇檔案來源",
        ["📂 本地資料夾", "📤 上傳檔案"],
        help="選擇要從本地資料夾載入或上傳新檔案"
    )
    
    selected_file_path = None
    
    if file_source == "📂 本地資料夾":
        # 本地資料夾路徑
        data_folder = "./test_data"
        
        # 確保資料夾存在
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            st.info(f"已建立資料夾：{data_folder}")
        
        # 獲取資料夾中的Excel檔案
        excel_files = [f for f in os.listdir(data_folder) 
                      if f.endswith(('.xlsx', '.xls')) and not f.startswith('~')]
        
        if excel_files:
            selected_file = st.selectbox(
                "選擇要評估的檔案",
                excel_files,
                help="從 test_data 資料夾中選擇檔案"
            )
            selected_file_path = os.path.join(data_folder, selected_file)
            
            # 顯示檔案資訊
            file_info = os.stat(selected_file_path)
            st.info(f"檔案大小：{file_info.st_size / 1024:.1f} KB")
        else:
            st.warning("⚠️ test_data 資料夾中沒有找到 Excel 檔案")
            st.markdown("""
            請將 Excel 檔案放入以下路徑：
            ```
            ./test_data/
            ```
            """)
    
    else:  # 上傳檔案
        uploaded_file = st.file_uploader(
            "選擇測試結果Excel檔案",
            type=['xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            # 保存上傳的檔案
            selected_file_path = "temp_uploaded.xlsx"
            with open(selected_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    # 執行評估按鈕
    if selected_file_path:
        if st.button("🚀 執行評估", type="primary", use_container_width=True):
            with st.spinner("評估中..."):
                evaluator = RAGEvaluator(selected_file_path)
                results = evaluator.evaluate_all()
                stats = evaluator.generate_summary_stats()
                
                # 保存結果到session state
                st.session_state['results'] = results
                st.session_state['stats'] = stats
                st.session_state['evaluator'] = evaluator
                st.session_state['file_name'] = os.path.basename(selected_file_path)
            
            st.success("✅ 評估完成！")

# 主要內容區
if 'results' in st.session_state:
    results = st.session_state['results']
    stats = st.session_state['stats']
    evaluator = st.session_state['evaluator']
    
    # 顯示正在評估的檔案
    if 'file_name' in st.session_state:
        st.info(f"📊 正在評估檔案：**{st.session_state['file_name']}**")
    
    # 建立頁籤
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 總覽", "📈 詳細評分", "🔍 問題分析", "📉 比較分析", "💾 下載結果"]
    )
    
    with tab1:
        st.header("評估總覽")
        
        # 顯示四種方法的整體表現
        methods = ['向量知識庫（原始版）', '向量知識庫（彙整版）', '智慧文檔知識庫（原始版）', '智慧文檔知識庫（彙整版）']
        method_display_names = ['向量知識庫（原始版）', '向量知識庫（彙整版）', '智慧文檔知識庫（原始版）', '智慧文檔知識庫（彙整版）']
        
        # 建立綜合指標表格
        st.markdown("### 📊 評估指標總覽")
        st.info(f"**資料筆數**: {len(results)} 筆測試題目")
        
        # 準備表格數據
        table_data = []
        for method in methods:
            # 計算高覆蓋率的題數
            high_coverage_count = (results[f'SCORE_{methods.index(method)+1}'] >= 80).sum()
            total_count = len(results)
            
            row = {
                '評估方法': method.replace('_', ' '),
                '🎯 平均覆蓋率': f"{stats[method]['平均覆蓋率']:.1f}%",
                '高覆蓋率比例 ℹ️': f"{stats[method]['高覆蓋率比例']:.1f}%",
                '🎭 平均忠誠度': f"{stats[method]['平均忠誠度']:.1f}%",
                '完全忠實比例': f"{stats[method]['完全忠實比例']:.1f}%",
                '📊 平均綜合評分': f"{stats[method]['平均綜合評分']:.1f}%"
            }
            table_data.append(row)
        
        # 建立DataFrame
        metrics_df = pd.DataFrame(table_data)
        
        # 使用st.table顯示（固定格式）或st.dataframe（可互動）
        # 設定樣式
        def highlight_best(s):
            if s.name == '🎯 平均覆蓋率' or s.name == '高覆蓋率比例 ℹ️' or s.name == '📊 平均綜合評分':
                # 這些指標越高越好
                is_max = s == s.max()
                return ['background-color: #2ECC71' if v else '' for v in is_max]
            elif s.name == '🎭 平均忠誠度':
                # 忠誠度越高越好
                is_max = s == s.max()
                return ['background-color: #2ECC71' if v else '' for v in is_max]
            elif s.name == '完全忠實比例':
                # 無幻覺比例越高越好
                is_max = s == s.max()
                return ['background-color: #2ECC71' if v else '' for v in is_max]
            return ['' for _ in s]
        
        # 轉換為數值以便比較（用於樣式）
        styled_df = metrics_df.copy()
        for col in ['🎯 平均覆蓋率', '高覆蓋率比例 ℹ️', '🎭 平均忠誠度', '完全忠實比例', '📊 平均綜合評分']:
            styled_df[col] = styled_df[col].str.replace('%', '').astype(float)
        
        # 應用樣式並顯示
        st.dataframe(
            styled_df.style.apply(highlight_best, axis=0).format({
                '🎯 平均覆蓋率': '{:.1f}%',
                '高覆蓋率比例 ℹ️': '{:.1f}%',
                '🎭 平均忠誠度': '{:.1f}%',
                '完全忠實比例': '{:.1f}%',
                '📊 平均綜合評分': '{:.1f}%'
            }),
            hide_index=True,
            use_container_width=True
        )
        
        # 在表格下方添加高覆蓋率比例的詳細說明
        st.markdown("#### 💡 高覆蓋率比例計算說明")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.info("**計算公式**")
        with col2:
            st.code("高覆蓋率比例 = (覆蓋率 ≥ 80% 的題數) ÷ 總題數 × 100%", language="text")
        
        # 顯示每個方法的詳細計算
        with st.expander("🔍 查看詳細計算過程"):
            for i, method in enumerate(methods):
                high_coverage_count = (results[f'SCORE_{i+1}'] >= 80).sum()
                total_count = len(results)
                percentage = stats[method]['高覆蓋率比例']
                
                st.markdown(f"**{method}**")
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(f"- 總題數：{total_count} 題")
                    st.markdown(f"- 高覆蓋率題數：{high_coverage_count} 題")
                with col_b:
                    st.markdown(f"- 計算：{high_coverage_count}/{total_count} × 100%")
                    st.markdown(f"- 結果：**{percentage:.1f}%**")
                st.markdown("---")
        
        # 添加說明
        with st.expander("📖 所有指標說明"):
            st.markdown("""
            - **🎯 平均覆蓋率**: 回答中包含應回答詞彙的比例（越高越好）
            - **高覆蓋率比例 ℹ️**: 覆蓋率≥80%的問題佔比（越高越好）
                - 此指標反映系統在多少比例的問題上能達到優秀表現
                - 例如：46.7% 表示有 46.7% 的問題達到 80% 以上的覆蓋率
            - **🎭 平均忠誠度**: AI回答忠實於原始資料的程度（越高越好）
            - **完全忠實比例**: 完全不虛構內容的問題佔比（越高越好）
            - **📊 平均綜合評分**: 覆蓋率 × 0.5 + 忠誠度 × 0.5（越高越好）
            
            🟢 綠色標記代表該指標在四種方法中表現最佳
            """)
        
        # 添加關鍵發現
        st.markdown("### 🔍 關鍵發現")
        
        # 找出最佳方法
        best_coverage_method = max(stats.items(), key=lambda x: x[1]['平均覆蓋率'])[0]
        best_faithfulness_method = max(stats.items(), key=lambda x: x[1]['平均忠誠度'])[0]
        best_overall_method = max(stats.items(), key=lambda x: x[1]['平均綜合評分'])[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**最佳覆蓋率**: {best_coverage_method}")
        with col2:
            st.success(f"**最高忠誠度**: {best_faithfulness_method}")
        with col3:
            st.warning(f"**最佳綜合**: {best_overall_method}")
        
        # 分隔線
        st.markdown("---")
        
        # 建立兩個並排的圖表
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        col_left, col_right = st.columns(2)
        
        with col_left:
            # 覆蓋率對比圖
            st.subheader("📊 覆蓋率對比分析")
            
            coverage_data = []
            for method in methods:
                coverage_data.append({
                    '方法': method,
                    '平均覆蓋率': stats[method]['平均覆蓋率'],
                    '高覆蓋率比例': stats[method]['高覆蓋率比例']
                })
            
            coverage_df = pd.DataFrame(coverage_data)
            
            # 建立分組柱狀圖
            fig_coverage = go.Figure()
            
            x = list(range(len(methods)))
            width = 0.35
            
            fig_coverage.add_trace(go.Bar(
                x=[i - width/2 for i in x],
                y=coverage_df['平均覆蓋率'],
                name='平均覆蓋率',
                text=[f'{v:.1f}%' for v in coverage_df['平均覆蓋率']],
                textposition='auto',
                marker_color='#3498DB'
            ))
            
            fig_coverage.add_trace(go.Bar(
                x=[i + width/2 for i in x],
                y=coverage_df['高覆蓋率比例'],
                name='高覆蓋率比例 (≥80%)',
                text=[f'{v:.1f}%' for v in coverage_df['高覆蓋率比例']],
                textposition='auto',
                marker_color='#2ECC71'
            ))
            
            fig_coverage.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=x,
                    ticktext=['向量\n(原始)', '向量\n(彙整)', '智慧文檔\n(原始)', '智慧文檔\n(彙整)'],
                    tickangle=-45
                ),
                yaxis=dict(title='百分比 (%)'),
                barmode='group',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_coverage, use_container_width=True)
        
        with col_right:
            # 忠誠度對比圖
            st.subheader("📊 忠誠度對比分析")
            
            faithfulness_data = []
            for i, method in enumerate(methods):
                faithfulness_data.append({
                    '方法': method,
                    '平均忠誠度': stats[method]['平均忠誠度'],
                    '完全忠實比例': stats[method]['完全忠實比例'],
                    '顏色': colors[i]
                })
            
            faithfulness_df = pd.DataFrame(faithfulness_data)
            
            # 忠誠度柱狀圖
            fig_faithfulness = go.Figure()
            
            # 使用顏色編碼顯示忠誠度
            bar_colors = []
            for score in faithfulness_df['平均忠誠度']:
                if score >= 90:
                    bar_colors.append('#2ECC71')  # 綠色：優秀
                elif score >= 70:
                    bar_colors.append('#F39C12')  # 黃色：良好
                elif score >= 50:
                    bar_colors.append('#E67E22')  # 橙色：一般
                else:
                    bar_colors.append('#E74C3C')  # 紅色：不佳
            
            fig_faithfulness.add_trace(
                go.Bar(
                    x=faithfulness_df['方法'],
                    y=faithfulness_df['平均忠誠度'],
                    name='平均忠誠度',
                    text=[f'{v:.1f}%' for v in faithfulness_df['平均忠誠度']],
                    textposition='outside',
                    marker_color=bar_colors
                )
            )
            
            # 添加完全忠實比例作為文字標籤
            for i, row in faithfulness_df.iterrows():
                fig_faithfulness.add_annotation(
                    x=row['方法'],
                    y=row['平均忠誠度'] + 2,
                    text=f"完全忠實: {row['完全忠實比例']:.1f}%",
                    showarrow=False,
                    font=dict(size=10, color='gray')
                )
            
            fig_faithfulness.update_xaxes(tickangle=-45)
            fig_faithfulness.update_yaxes(title_text="忠誠度 (%)")
            fig_faithfulness.update_layout(
                height=400,
                showlegend=False,
                yaxis_range=[0, 110]  # 留空間給標註
            )
            
            st.plotly_chart(fig_faithfulness, use_container_width=True)
    
    with tab2:
        st.header("詳細評分分析")
        st.info(f"**總題數**: {len(results)} 題")
        
        # 選擇要查看的方法
        selected_method = st.selectbox(
            "選擇評估方法",
            options=['全部'] + methods
        )
        
        # 建立評分分佈圖
        if selected_method == '全部':
            # 建立子圖
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=methods,
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for i, method in enumerate(methods):
                row = i // 2 + 1
                col = i % 2 + 1
                
                # 覆蓋率分佈
                fig.add_trace(
                    go.Histogram(
                        x=results[f'SCORE_{i+1}'],
                        name=f'{method} 覆蓋率',
                        nbinsx=10,  # 減少組數讓分佈更清楚
                        marker_color=colors[i],
                        showlegend=False,
                        xbins=dict(
                            start=0,
                            end=100,
                            size=10  # 每10%一組
                        )
                    ),
                    row=row, col=col
                )
                
                # 更新子圖的軸標籤
                fig.update_xaxes(title_text="覆蓋率 (%)", row=row, col=col)
                fig.update_yaxes(title_text="題數", row=row, col=col)
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 單一方法的詳細分析
            method_idx = methods.index(selected_method) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 覆蓋率分佈
                fig1 = px.histogram(
                    results,
                    x=f'SCORE_{method_idx}',
                    nbins=20,
                    title=f"{selected_method} - 覆蓋率分佈",
                    labels={f'SCORE_{method_idx}': '覆蓋率 (%)'},
                    color_discrete_sequence=[colors[method_idx-1]]
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # 忠誠度分佈
                fig2 = px.pie(
                    results[f'FAITHFULNESS_DESC_{method_idx}'].value_counts().reset_index(),
                    values='count',
                    names='FAITHFULNESS_DESC_' + str(method_idx),
                    title=f"{selected_method} - 忠誠度類型分佈"
                )
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.header("問題層級分析")
        
        # 建立問題評分表格
        display_columns = ['序號', '測試問題']
        for i in range(1, 5):
            display_columns.extend([f'SCORE_{i}', f'FAITHFULNESS_{i}', f'TOTAL_SCORE_{i}'])
        
        # 建立可排序的表格
        st.dataframe(
            results[display_columns],
            use_container_width=True,
            height=400
        )
        
        # 找出表現最差的問題
        st.subheader("🚨 需要關注的問題")
        
        # 計算每個問題的平均綜合評分
        results['avg_total_score'] = results[[f'TOTAL_SCORE_{i}' for i in range(1, 5)]].mean(axis=1)
        
        worst_questions = results.nsmallest(5, 'avg_total_score')[
            ['序號', '測試問題', 'avg_total_score']
        ]
        
        st.dataframe(worst_questions, use_container_width=True)
    
    with tab4:
        st.header("方法比較分析")
        
        # 建立箱型圖比較
        comparison_data = []
        for i, method in enumerate(methods):
            for score in results[f'SCORE_{i+1}']:
                comparison_data.append({
                    '方法': method,
                    '覆蓋率': score,
                    '類型': '覆蓋率'
                })
            
            for score in results[f'TOTAL_SCORE_{i+1}']:
                comparison_data.append({
                    '方法': method,
                    '綜合評分': score,
                    '類型': '綜合評分'
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 箱型圖
        fig = px.box(
            comparison_df[comparison_df['類型'] == '覆蓋率'],
            x='方法',
            y='覆蓋率',
            title="覆蓋率分佈比較",
            color='方法',
            color_discrete_sequence=colors
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 建立熱力圖
        st.subheader("📊 問題-方法表現熱力圖")
        
        heatmap_data = results[[f'TOTAL_SCORE_{i}' for i in range(1, 5)]].values
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=methods,
            y=[f"問題 {i}" for i in results['序號']],
            colorscale='RdYlGn',
            text=np.round(heatmap_data, 1),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            height=800,
            xaxis_title="評估方法",
            yaxis_title="測試問題"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("下載評估結果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 生成詳細報告", type="primary"):
                # 生成報告
                from datetime import datetime
                output_path = f'RAG評估結果_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                evaluator.save_results(output_path)
                
                st.success(f"✅ 報告已生成: {output_path}")
                
                # 提供下載
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 下載評估報告",
                        data=f,
                        file_name=output_path,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
        
        with col2:
            # 顯示統計摘要
            st.subheader("📈 統計摘要")
            
            summary_data = []
            for method, method_stats in stats.items():
                row = {'方法': method}
                row.update(method_stats)
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.round(2), use_container_width=True)

else:
    # 沒有上傳檔案時的提示
    st.info("👈 請從側邊欄上傳測試結果Excel檔案開始評估")
    
    # 顯示使用說明
    with st.expander("📖 使用說明"):
        st.markdown("""
        ### 評分方法說明
        
        #### 1. **覆蓋率評分 (SCORE_1~4)**
        - 計算回答中包含多少「應回答之詞彙」中的關鍵詞
        - 評分 = (匹配到的關鍵詞數 / 總關鍵詞數) × 100
        
        #### 2. **忠誠度評分 (FAITHFULNESS_1~4)**
        - **100分**: 完全忠實 - 完全基於原始資料
        - **90分**: 極高忠實 - 添加合理解釋
        - **75分**: 高度忠實 - 包含少量額外數據
        - **50分**: 中度忠實 - 包含多個未提及的具體數據
        - **0分**: 不忠實 - 包含大量虛構資訊
        
        #### 3. **綜合評分 (TOTAL_SCORE_1~4)**
        - 綜合評分 = 覆蓋率 × 0.5 + 忠誠度 × 0.5
        - 兩個指標各占50%的權重
        
        ### 資料處理方式說明
        - **向量知識庫（原始版）**: 原始資料直接向量化
        - **向量知識庫（彙整版）**: 彙整後的資料向量化
        - **智慧文檔知識庫（原始版）**: 原始資料智慧文檔處理
        - **智慧文檔知識庫（彙整版）**: 彙整後的資料智慧文檔處理
        """)

# 頁尾
st.markdown("---")
st.caption("RAG LLM 評估系統 v1.0 | 開發日期: 2025")