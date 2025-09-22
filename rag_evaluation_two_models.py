import pandas as pd
import re
import jieba
import numpy as np
from typing import List, Dict, Tuple
import streamlit as st
from datetime import datetime

class RAGEvaluatorTwoModels:
    """專門用於比較兩個知識庫版本的評估器：原始版本 vs 彙整版本"""
    
    def __init__(self, excel_path: str, model_type: str = "vector"):
        """
        初始化RAG評估器 - 兩個模型比較版
        
        參數:
            excel_path: Excel或CSV檔案路徑
            model_type: "vector" (向量知識庫) 或 "smart_doc" (智慧文檔知識庫)
        """
        # 根據檔案類型讀取資料
        if excel_path.lower().endswith('.csv'):
            self.df = pd.read_csv(excel_path, encoding='utf-8-sig')
        else:
            self.df = pd.read_excel(excel_path)
        self.model_type = model_type
        jieba.setLogLevel(20)
        
        # 檢查可用的欄位
        available_columns = self.df.columns.tolist()
        print(f"偵測到的欄位: {available_columns}")
        
        # 自動偵測欄位名稱（處理可能的變體）
        vector_original_variants = ['向量知識庫（原始版）', '向量知識庫(原始版)', '向量知識庫_原始版', 'Vector_Original']
        vector_optimized_variants = ['向量知識庫（彙整版）', '向量知識庫(彙整版)', '向量知識庫_彙整版', 'Vector_Optimized']
        smart_original_variants = ['智慧文檔知識庫（原始版）', '智慧文檔知識庫(原始版)', '智慧文檔知識庫_原始版', 'SmartDoc_Original']
        smart_optimized_variants = ['智慧文檔知識庫（彙整版）', '智慧文檔知識庫(彙整版)', '智慧文檔知識庫_彙整版', 'SmartDoc_Optimized']
        
        # 新增跨技術比較模式
        if model_type == "cross":
            # 跨技術比較：向量原始 vs 智慧文檔彙整
            self.original_col = None
            self.optimized_col = None
            
            # 尋找向量原始版
            for col in available_columns:
                if '向量' in col and ('原始' in col or 'Original' in col):
                    self.original_col = col
                    break
            
            # 尋找智慧文檔彙整版
            for col in available_columns:
                if ('智慧' in col or '文檔' in col) and ('彙整' in col or 'Optimized' in col):
                    self.optimized_col = col
                    break
            
            self.model_name = "跨技術比較"
            
            if not self.original_col or not self.optimized_col:
                raise ValueError(f"無法找到跨技術比較所需的欄位。需要向量原始版和智慧文檔彙整版。可用欄位: {available_columns}")
        
        # 根據model_type尋找對應的欄位
        elif model_type == "vector":
            # 尋找向量知識庫欄位
            self.original_col = None
            self.optimized_col = None
            
            for col in vector_original_variants:
                if col in available_columns:
                    self.original_col = col
                    break
                    
            for col in vector_optimized_variants:
                if col in available_columns:
                    self.optimized_col = col
                    break
                    
            self.model_name = "向量知識庫"
            
            # 如果找不到預設的欄位名稱，嘗試自動偵測
            if not self.original_col or not self.optimized_col:
                # 尋找包含"向量"和"原始"的欄位
                vector_columns = [col for col in available_columns if '向量' in col or 'vector' in col.lower()]
                if len(vector_columns) >= 2:
                    # 假設第一個是原始版，第二個是彙整版
                    self.original_col = vector_columns[0]
                    self.optimized_col = vector_columns[1]
                    print(f"自動選擇欄位: {self.original_col} vs {self.optimized_col}")
                else:
                    raise ValueError(f"無法找到向量知識庫相關欄位。可用欄位: {available_columns}")
                    
        else:  # smart_doc
            # 尋找智慧文檔知識庫欄位
            self.original_col = None
            self.optimized_col = None
            
            for col in smart_original_variants:
                if col in available_columns:
                    self.original_col = col
                    break
                    
            for col in smart_optimized_variants:
                if col in available_columns:
                    self.optimized_col = col
                    break
                    
            self.model_name = "智慧文檔知識庫"
            
            # 如果找不到預設的欄位名稱，嘗試自動偵測
            if not self.original_col or not self.optimized_col:
                # 尋找包含"智慧"或"文檔"的欄位
                smart_columns = [col for col in available_columns if '智慧' in col or '文檔' in col or 'smart' in col.lower() or 'doc' in col.lower()]
                if len(smart_columns) >= 2:
                    # 假設第一個是原始版，第二個是彙整版
                    self.original_col = smart_columns[0]
                    self.optimized_col = smart_columns[1]
                    print(f"自動選擇欄位: {self.original_col} vs {self.optimized_col}")
                else:
                    raise ValueError(f"無法找到智慧文檔知識庫相關欄位。可用欄位: {available_columns}")
        
        print(f"使用欄位 - 原始: {self.original_col}, 優化: {self.optimized_col}")
            
    def extract_keywords(self, text: str) -> List[str]:
        """從應回答詞彙中提取關鍵詞"""
        if pd.isna(text):
            return []
        
        # 移除編號和標點符號
        text = re.sub(r'\d+\.', '', text)
        text = re.sub(r'[：:。，,、\(\)]', ' ', text)
        
        # 提取關鍵詞
        keywords = []
        
        # 保留完整的專有名詞
        special_terms = [
            "工作許可證", "施工轄區", "包商名稱", "作業內容",
            "承包商現場負責人", "工安業務主管", "施工人員",
            "煙火管制區", "電焊", "切割", "烘烤"
        ]
        
        for term in special_terms:
            if term in text:
                keywords.append(term)
                text = text.replace(term, " ")
        
        # 使用jieba分詞處理剩餘文字
        words = jieba.cut(text)
        for word in words:
            if len(word.strip()) > 1 and word.strip() not in keywords:
                keywords.append(word.strip())
        
        return keywords
    
    def calculate_coverage_score(self, answer: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """計算關鍵詞覆蓋率評分"""
        if pd.isna(answer) or not keywords:
            return 0.0, []
        
        matched_keywords = []
        answer_lower = answer.lower()
        
        for keyword in keywords:
            if keyword.lower() in answer_lower:
                matched_keywords.append(keyword)
            elif self._is_similar_term(keyword, answer):
                matched_keywords.append(keyword)
        
        coverage_rate = len(matched_keywords) / len(keywords) if keywords else 0
        return coverage_rate * 100, matched_keywords
    
    def _is_similar_term(self, keyword: str, answer: str) -> bool:
        """檢查是否有相似詞彙"""
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
        """評估AI回答的忠誠度"""
        if pd.isna(answer):
            return 100, "無回答（無虛構風險）", {}
        
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
        
        # 評估忠誠度
        faithfulness_score = 100
        faithfulness_type = "完全忠實"
        
        # 檢查解釋性詞彙
        explanation_words = ["因此", "所以", "包括", "例如", "如", "即", "也就是", "用於", "目的"]
        explanation_count = sum(1 for word in explanation_words if word in answer)
        
        # 根據額外內容扣分
        if len(analysis["extra_numbers"]) > 2 or len(analysis["extra_dates"]) > 1:
            faithfulness_score = 50
            faithfulness_type = "中度忠實：包含多個未提及的具體數據"
        elif len(analysis["extra_numbers"]) > 0 or len(analysis["extra_dates"]) > 0:
            faithfulness_score = 75
            faithfulness_type = "高度忠實：包含少量額外數據"
        elif explanation_count > 3:
            faithfulness_score = 90
            faithfulness_type = "極高忠實：添加合理解釋"
        else:
            faithfulness_score = 100
            faithfulness_type = "完全忠實：完全基於原始資料"
        
        return faithfulness_score, faithfulness_type, analysis
    
    def evaluate_all(self) -> pd.DataFrame:
        """執行完整評估 - 只評估兩個版本"""
        # 添加評估欄位
        # 原始版本
        self.df['SCORE_ORIGINAL'] = 0.0
        self.df['FAITHFULNESS_ORIGINAL'] = 100
        self.df['FAITHFULNESS_DESC_ORIGINAL'] = ""
        self.df['MATCHED_KEYWORDS_ORIGINAL'] = ""
        self.df['ANSWER_ORIGINAL'] = self.df[self.original_col]
        
        # 優化版本
        self.df['SCORE_OPTIMIZED'] = 0.0
        self.df['FAITHFULNESS_OPTIMIZED'] = 100
        self.df['FAITHFULNESS_DESC_OPTIMIZED'] = ""
        self.df['MATCHED_KEYWORDS_OPTIMIZED'] = ""
        self.df['ANSWER_OPTIMIZED'] = self.df[self.optimized_col]
        
        # 對每一行進行評估
        for idx, row in self.df.iterrows():
            keywords = self.extract_keywords(row['應回答之詞彙'])
            
            # 評估原始版本
            coverage_score_orig, matched_orig = self.calculate_coverage_score(
                row[self.original_col], keywords
            )
            self.df.at[idx, 'SCORE_ORIGINAL'] = coverage_score_orig
            self.df.at[idx, 'MATCHED_KEYWORDS_ORIGINAL'] = ', '.join(matched_orig)
            
            faithfulness_score_orig, faithfulness_desc_orig, _ = self.evaluate_faithfulness(
                row[self.original_col], keywords, row['測試問題']
            )
            self.df.at[idx, 'FAITHFULNESS_ORIGINAL'] = faithfulness_score_orig
            self.df.at[idx, 'FAITHFULNESS_DESC_ORIGINAL'] = faithfulness_desc_orig
            
            # 評估優化版本
            coverage_score_opt, matched_opt = self.calculate_coverage_score(
                row[self.optimized_col], keywords
            )
            self.df.at[idx, 'SCORE_OPTIMIZED'] = coverage_score_opt
            self.df.at[idx, 'MATCHED_KEYWORDS_OPTIMIZED'] = ', '.join(matched_opt)
            
            faithfulness_score_opt, faithfulness_desc_opt, _ = self.evaluate_faithfulness(
                row[self.optimized_col], keywords, row['測試問題']
            )
            self.df.at[idx, 'FAITHFULNESS_OPTIMIZED'] = faithfulness_score_opt
            self.df.at[idx, 'FAITHFULNESS_DESC_OPTIMIZED'] = faithfulness_desc_opt
        
        # 計算綜合評分（覆蓋率和忠誠度各占50%）
        self.df['TOTAL_SCORE_ORIGINAL'] = (
            self.df['SCORE_ORIGINAL'] * 0.5 + 
            self.df['FAITHFULNESS_ORIGINAL'] * 0.5
        )
        self.df['TOTAL_SCORE_OPTIMIZED'] = (
            self.df['SCORE_OPTIMIZED'] * 0.5 + 
            self.df['FAITHFULNESS_OPTIMIZED'] * 0.5
        )
        
        # 計算改善幅度
        self.df['COVERAGE_IMPROVEMENT'] = self.df['SCORE_OPTIMIZED'] - self.df['SCORE_ORIGINAL']
        self.df['FAITHFULNESS_IMPROVEMENT'] = self.df['FAITHFULNESS_OPTIMIZED'] - self.df['FAITHFULNESS_ORIGINAL']
        self.df['TOTAL_IMPROVEMENT'] = self.df['TOTAL_SCORE_OPTIMIZED'] - self.df['TOTAL_SCORE_ORIGINAL']
        
        return self.df
    
    def save_results(self, output_path: str):
        """保存評估結果到Excel或CSV"""
        # 選擇要輸出的欄位
        output_columns = ['序號', '測試資料', '測試問題', '應回答之詞彙']
        
        # 添加評分相關欄位
        output_columns.extend([
            'SCORE_ORIGINAL', 'FAITHFULNESS_ORIGINAL', 'TOTAL_SCORE_ORIGINAL',
            'SCORE_OPTIMIZED', 'FAITHFULNESS_OPTIMIZED', 'TOTAL_SCORE_OPTIMIZED',
            'COVERAGE_IMPROVEMENT', 'FAITHFULNESS_IMPROVEMENT', 'TOTAL_IMPROVEMENT',
            'MATCHED_KEYWORDS_ORIGINAL', 'MATCHED_KEYWORDS_OPTIMIZED'
        ])
        
        # 創建輸出DataFrame
        output_df = self.df[output_columns].copy()
        
        # 根據檔案類型保存
        if output_path.lower().endswith('.csv'):
            # 儲存為CSV
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
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
                worksheet.set_column('E:M', 15)  # 評分欄位
                worksheet.set_column('N:O', 50)  # 關鍵詞匹配欄位
            
        print(f"評分結果已保存到: {output_path}")
    
    def generate_summary_stats(self) -> Dict:
        """生成統計摘要"""
        stats = {
            '原始版本': {
                '平均覆蓋率': self.df['SCORE_ORIGINAL'].mean(),
                '平均忠誠度': self.df['FAITHFULNESS_ORIGINAL'].mean(),
                '平均綜合評分': self.df['TOTAL_SCORE_ORIGINAL'].mean(),
                '完全忠實比例': (self.df['FAITHFULNESS_ORIGINAL'] == 100).sum() / len(self.df) * 100,
                '高覆蓋率比例': (self.df['SCORE_ORIGINAL'] >= 80).sum() / len(self.df) * 100,
                '高忠誠度比例': (self.df['FAITHFULNESS_ORIGINAL'] >= 90).sum() / len(self.df) * 100
            },
            '彙整優化版本': {
                '平均覆蓋率': self.df['SCORE_OPTIMIZED'].mean(),
                '平均忠誠度': self.df['FAITHFULNESS_OPTIMIZED'].mean(),
                '平均綜合評分': self.df['TOTAL_SCORE_OPTIMIZED'].mean(),
                '完全忠實比例': (self.df['FAITHFULNESS_OPTIMIZED'] == 100).sum() / len(self.df) * 100,
                '高覆蓋率比例': (self.df['SCORE_OPTIMIZED'] >= 80).sum() / len(self.df) * 100,
                '高忠誠度比例': (self.df['FAITHFULNESS_OPTIMIZED'] >= 90).sum() / len(self.df) * 100
            },
            '改善效果': {
                '平均覆蓋率提升': self.df['COVERAGE_IMPROVEMENT'].mean(),
                '平均忠誠度提升': self.df['FAITHFULNESS_IMPROVEMENT'].mean(),
                '平均綜合評分提升': self.df['TOTAL_IMPROVEMENT'].mean(),
                '顯著改善比例': (self.df['TOTAL_IMPROVEMENT'] >= 10).sum() / len(self.df) * 100,
                '效果退步比例': (self.df['TOTAL_IMPROVEMENT'] < 0).sum() / len(self.df) * 100
            }
        }
        
        return stats

# 使用範例
if __name__ == "__main__":
    # 評估向量知識庫
    evaluator_vector = RAGEvaluatorTwoModels('測試結果驗證.xlsx', model_type='vector')
    results_vector = evaluator_vector.evaluate_all()
    evaluator_vector.save_results(f'向量知識庫_比較結果_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    
    # 評估智慧文檔知識庫
    evaluator_smart = RAGEvaluatorTwoModels('測試結果驗證.xlsx', model_type='smart_doc')
    results_smart = evaluator_smart.evaluate_all()
    evaluator_smart.save_results(f'智慧文檔知識庫_比較結果_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    
    # 顯示統計
    print("\n向量知識庫 統計摘要:")
    stats_vector = evaluator_vector.generate_summary_stats()
    for category, metrics in stats_vector.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")