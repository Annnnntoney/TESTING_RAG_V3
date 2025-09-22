import pandas as pd
import re
import jieba
import numpy as np
from typing import List, Dict, Tuple
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class RAGEvaluatorV2:
    def __init__(self, excel_path: str):
        """初始化RAG評估器 - 改進版"""
        self.df = pd.read_excel(excel_path)
        jieba.setLogLevel(20)
        
    def extract_keywords(self, text: str) -> List[str]:
        """從應回答詞彙中提取關鍵詞 - 改進版"""
        if pd.isna(text):
            return []
        
        # 移除編號和標點符號
        text = re.sub(r'\d+\.', '', text)
        text = re.sub(r'[：:。，,、\(\)]', ' ', text)
        
        # 提取關鍵詞（改進版）
        keywords = []
        
        # 1. 保留完整的專有名詞
        # 例如：「承包商現場負責人」不應該被拆分
        special_terms = [
            "工作許可證", "施工轄區", "包商名稱", "作業內容",
            "承包商現場負責人", "工安業務主管", "施工人員",
            "煙火管制區", "電焊", "切割", "烘烤"
        ]
        
        for term in special_terms:
            if term in text:
                
                keywords.append(term)
                text = text.replace(term, " ")  # 避免重複提取
        
        # 2. 使用jieba分詞處理剩餘文字
        words = jieba.cut(text)
        for word in words:
            if len(word.strip()) > 1 and word.strip() not in keywords:
                keywords.append(word.strip())
        
        return keywords
    
    def calculate_coverage_score(self, answer: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """計算關鍵詞覆蓋率評分 - 改進版"""
        if pd.isna(answer) or not keywords:
            return 0.0, []
        
        matched_keywords = []
        answer_lower = answer.lower()
        
        for keyword in keywords:
            # 改進匹配邏輯，支援部分匹配
            if keyword.lower() in answer_lower:
                matched_keywords.append(keyword)
            # 處理可能的同義詞或相似詞
            elif self._is_similar_term(keyword, answer):
                matched_keywords.append(keyword)
        
        coverage_rate = len(matched_keywords) / len(keywords) if keywords else 0
        return coverage_rate * 100, matched_keywords
    
    def _is_similar_term(self, keyword: str, answer: str) -> bool:
        """檢查是否有相似詞彙"""
        # 定義同義詞對照表
        synonyms = {
            "包商": ["承包商", "廠商", "承攬商"],
            "負責人": ["主管", "管理人", "聯絡人"],
            "工安": ["安全", "職安", "工業安全"],
            "許可證": ["許可", "證明", "核准"],
        }
        
        answer_lower = answer.lower()
        for key, similar_terms in synonyms.items():
            if key in keyword:
                for term in similar_terms:
                    if term in answer_lower:
                        return True
        return False
    
    def evaluate_faithfulness(self, answer: str, reference_keywords: List[str], 
                             question: str) -> Tuple[float, str, Dict]:
        """評估AI回答的忠誠度
        
        返回: (忠誠度分數, 忠誠度說明, 詳細分析)
        忠誠度分數: 0-100，越高表示越忠實於原始資料
        """
        if pd.isna(answer):
            return 100, "無回答（無虛構風險）", {}
        
        # 分析回答內容
        analysis = {
            "extra_numbers": [],
            "extra_dates": [],
            "extra_specific_terms": [],
            "reasonable_explanations": 0,
            "total_sentences": 0
        }
        
        # 將回答分句
        sentences = re.split(r'[。！？\n]', answer)
        analysis["total_sentences"] = len([s for s in sentences if s.strip()])
        
        # 檢查數字和日期
        numbers_in_answer = re.findall(r'\b\d+\b', answer)
        dates_in_answer = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}', answer)
        
        # 檢查參考答案中的數字和日期
        reference_text = ' '.join(reference_keywords)
        numbers_in_ref = re.findall(r'\b\d+\b', reference_text)
        dates_in_ref = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}', reference_text)
        
        # 找出額外的數字和日期
        analysis["extra_numbers"] = [n for n in numbers_in_answer if n not in numbers_in_ref]
        analysis["extra_dates"] = [d for d in dates_in_answer if d not in dates_in_ref]
        
        # 評估忠誠度（改進版）
        # 忠誠度分數：100 = 完全忠實，0 = 完全虛構
        faithfulness_score = 100  # 從滿分開始扣分
        faithfulness_type = "完全忠實"
        
        # 1. 檢查是否有解釋性或連接性詞彙（這些通常是合理的）
        explanation_words = ["因此", "所以", "包括", "例如", "如", "即", "也就是", "用於", "目的"]
        explanation_count = sum(1 for word in explanation_words if word in answer)
        
        # 2. 根據額外內容扣分
        if len(analysis["extra_numbers"]) > 2 or len(analysis["extra_dates"]) > 1:
            faithfulness_score = 50  # 扣50分
            faithfulness_type = "中度忠實：包含多個未提及的具體數據"
        elif len(analysis["extra_numbers"]) > 0 or len(analysis["extra_dates"]) > 0:
            faithfulness_score = 75  # 扣25分
            faithfulness_type = "高度忠實：包含少量額外數據"
        elif explanation_count > 3:
            faithfulness_score = 90  # 扣10分
            faithfulness_type = "極高忠實：添加合理解釋"
        else:
            faithfulness_score = 100
            faithfulness_type = "完全忠實：完全基於原始資料"
        
        # 3. 檢查是否有明顯錯誤或虛構（嚴重違反忠誠度）
        # 這裡可以加入特定的錯誤檢測邏輯
        
        return faithfulness_score, faithfulness_type, analysis
    
    def evaluate_all(self) -> pd.DataFrame:
        """執行完整評估 - 改進版"""
        # 添加新欄位
        for i in range(1, 5):
            self.df[f'FAITHFULNESS_{i}'] = 100
            self.df[f'FAITHFULNESS_DESC_{i}'] = ""
            self.df[f'MATCHED_KEYWORDS_{i}'] = ""
        
        # 對每一行進行評估
        for idx, row in self.df.iterrows():
            keywords = self.extract_keywords(row['應回答之詞彙'])
            
            # 評估四種不同的處理方式
            columns = ['向量知識庫（原始版）', '向量知識庫（彙整版）', '智慧文檔知識庫（原始版）', '智慧文檔知識庫（彙整版）']
            score_columns = ['SCORE_1', 'SCORE_2', 'SCORE_3', 'SCORE_4']
            
            for i, (col, score_col) in enumerate(zip(columns, score_columns)):
                # 計算覆蓋率評分
                coverage_score, matched = self.calculate_coverage_score(row[col], keywords)
                self.df.at[idx, score_col] = coverage_score
                self.df.at[idx, f'MATCHED_KEYWORDS_{i+1}'] = ', '.join(matched)
                
                # 評估忠誠度
                faithfulness_score, faithfulness_desc, _ = self.evaluate_faithfulness(
                    row[col], keywords, row['測試問題']
                )
                self.df.at[idx, f'FAITHFULNESS_{i+1}'] = faithfulness_score
                self.df.at[idx, f'FAITHFULNESS_DESC_{i+1}'] = faithfulness_desc
        
        # 計算綜合評分（新公式：覆蓋率和忠誠度各占50%）
        for i in range(1, 5):
            self.df[f'TOTAL_SCORE_{i}'] = (
                self.df[f'SCORE_{i}'] * 0.5 + 
                self.df[f'FAITHFULNESS_{i}'] * 0.5
            )
        
        return self.df
    
    def save_results(self, output_path: str):
        """保存評估結果到Excel - 改進版"""
        # 選擇要輸出的欄位
        output_columns = ['序號', '測試資料', '測試問題', '應回答之詞彙']
        
        # 添加各項評分和匹配詳情
        for i in range(1, 5):
            output_columns.extend([
                f'SCORE_{i}', 
                f'FAITHFULNESS_{i}', 
                f'TOTAL_SCORE_{i}',
                f'MATCHED_KEYWORDS_{i}'
            ])
        
        # 創建輸出DataFrame
        output_df = self.df[output_columns].copy()
        
        # 寫入Excel，添加格式
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            output_df.to_excel(writer, sheet_name='評分結果', index=False)
            
            # 獲取工作簿和工作表
            workbook = writer.book
            worksheet = writer.sheets['評分結果']
            
            # 設定格式
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D7E4BD',
                'border': 1
            })
            
            # 設定列寬
            worksheet.set_column('A:A', 8)   # 序號
            worksheet.set_column('B:B', 20)  # 測試資料
            worksheet.set_column('C:C', 40)  # 測試問題
            worksheet.set_column('D:D', 50)  # 應回答之詞彙
            worksheet.set_column('E:T', 15)  # 評分欄位
            
        print(f"評分結果已保存到: {output_path}")
    
    def generate_summary_stats(self) -> Dict:
        """生成統計摘要"""
        stats = {}
        
        for i in range(1, 5):
            method_name = ['向量知識庫（原始版）', '向量知識庫（彙整版）', 
                          '智慧文檔知識庫（原始版）', '智慧文檔知識庫（彙整版）'][i-1]
            
            stats[method_name] = {
                '平均覆蓋率': self.df[f'SCORE_{i}'].mean(),
                '平均忠誠度': self.df[f'FAITHFULNESS_{i}'].mean(),
                '平均綜合評分': self.df[f'TOTAL_SCORE_{i}'].mean(),
                '完全忠實比例': (self.df[f'FAITHFULNESS_{i}'] == 100).sum() / len(self.df) * 100,
                '高覆蓋率比例': (self.df[f'SCORE_{i}'] >= 80).sum() / len(self.df) * 100,
                '高忠誠度比例': (self.df[f'FAITHFULNESS_{i}'] >= 90).sum() / len(self.df) * 100
            }
        
        return stats

# 使用範例
if __name__ == "__main__":
    evaluator = RAGEvaluatorV2('測試結果驗證.xlsx')
    results = evaluator.evaluate_all()
    evaluator.save_results('RAG評估結果_V2_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.xlsx')
    stats = evaluator.generate_summary_stats()
    print("\n統計摘要 (改進版):")
    for method, method_stats in stats.items():
        print(f"\n{method}:")
        for metric, value in method_stats.items():
            print(f"  {metric}: {value:.2f}")