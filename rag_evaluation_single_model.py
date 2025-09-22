import pandas as pd
import re
import jieba
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

class RAGEvaluatorSingleModel:
    """單一模型評估器 - 適用於只有一個AI回答的測試檔案"""
    
    def __init__(self, excel_path: str):
        """初始化評估器"""
        self.df = pd.read_csv(excel_path) if excel_path.endswith('.csv') else pd.read_excel(excel_path)
        jieba.setLogLevel(20)
        
    def extract_keywords(self, text: str) -> List[str]:
        """從回答重點中提取關鍵詞"""
        if pd.isna(text):
            return []
        
        # 移除編號和標點符號
        text = re.sub(r'\d+\.', '', text)
        text = re.sub(r'[：:。，,、\(\)]', ' ', text)
        
        # 提取關鍵詞
        keywords = []
        
        # 保留完整的專有名詞
        special_terms = [
            "職業災害", "通報", "勞動檢查機構", "死亡災害",
            "永久全失能", "住院治療", "8小時", "職業安全衛生",
            "職業病", "職業傷害", "復工", "補助"
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
            "職災": ["職業災害", "職業傷害", "工傷"],
            "通報": ["報告", "申報", "告知"],
            "補助": ["補貼", "津貼", "給付"],
            "復工": ["返回工作", "重返職場", "回到崗位"]
        }
        
        answer_lower = answer.lower()
        for key, similar_terms in synonyms.items():
            if key in keyword:
                for term in similar_terms:
                    if term in answer_lower:
                        return True
        return False
    
    def evaluate_faithfulness(self, answer: str, keywords: List[str]) -> Tuple[float, str]:
        """評估回答的忠誠度 - 使用與原始系統相同的算法"""
        if pd.isna(answer):
            return 100, "無回答（無虛構風險）"
        
        # 分析回答內容
        analysis = {
            "extra_numbers": [],
            "extra_dates": [],
            "reasonable_explanations": 0
        }
        
        # 將回答分句
        sentences = re.split(r'[。！？\n]', answer)
        total_sentences = len([s for s in sentences if s.strip()])
        
        # 檢查數字和日期
        numbers_in_answer = re.findall(r'\b\d+\b', answer)
        dates_in_answer = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}', answer)
        
        # 檢查參考答案中的數字和日期
        reference_text = ' '.join(keywords)
        numbers_in_ref = re.findall(r'\b\d+\b', reference_text)
        dates_in_ref = re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}', reference_text)
        
        # 找出額外的數字和日期
        analysis["extra_numbers"] = [n for n in numbers_in_answer if n not in numbers_in_ref]
        analysis["extra_dates"] = [d for d in dates_in_answer if d not in dates_in_ref]
        
        # 評估忠誠度
        faithfulness_score = 100  # 從滿分開始
        
        # 檢查是否有解釋性詞彙（這些通常是合理的）
        explanation_words = ["因此", "所以", "包括", "例如", "如", "即", "也就是", "用於", "目的"]
        explanation_count = sum(1 for word in explanation_words if word in answer)
        
        # 根據額外內容扣分
        if len(analysis["extra_numbers"]) > 2 or len(analysis["extra_dates"]) > 1:
            faithfulness_score = 50
            desc = "中度忠實：包含多個未提及的具體數據"
        elif len(analysis["extra_numbers"]) > 0 or len(analysis["extra_dates"]) > 0:
            faithfulness_score = 75
            desc = "高度忠實：包含少量額外數據"
        elif explanation_count > 3:
            faithfulness_score = 90
            desc = "極高忠實：添加合理解釋"
        else:
            faithfulness_score = 100
            desc = "完全忠實：完全基於原始資料"
        
        return faithfulness_score, desc
    
    def evaluate_all(self) -> pd.DataFrame:
        """執行評估"""
        # 添加評估欄位
        self.df['關鍵詞列表'] = ""
        self.df['覆蓋率分數'] = 0
        self.df['匹配關鍵詞'] = ""
        self.df['忠誠度分數'] = 100
        self.df['忠誠度描述'] = ""
        self.df['綜合評分'] = 0
        
        # 對每一行進行評估
        for idx, row in self.df.iterrows():
            # 提取關鍵詞
            keywords = self.extract_keywords(row['回答重點'])
            self.df.at[idx, '關鍵詞列表'] = ', '.join(keywords)
            
            # 計算覆蓋率
            coverage_score, matched = self.calculate_coverage_score(row['UPGPT回答'], keywords)
            self.df.at[idx, '覆蓋率分數'] = coverage_score
            self.df.at[idx, '匹配關鍵詞'] = ', '.join(matched)
            
            # 評估忠誠度（使用與原始系統相同的方法）
            faithfulness_score, faithfulness_desc = self.evaluate_faithfulness(
                row['UPGPT回答'], keywords
            )
            self.df.at[idx, '忠誠度分數'] = faithfulness_score
            self.df.at[idx, '忠誠度描述'] = faithfulness_desc
            
            # 計算綜合評分
            self.df.at[idx, '綜合評分'] = (coverage_score * 0.5 + faithfulness_score * 0.5)
        
        return self.df
    
    def generate_summary(self) -> Dict:
        """生成評估摘要"""
        summary = {
            '總題數': len(self.df),
            '平均覆蓋率': self.df['覆蓋率分數'].mean(),
            '平均忠誠度': self.df['忠誠度分數'].mean(),
            '平均綜合評分': self.df['綜合評分'].mean(),
            '高覆蓋率題數': (self.df['覆蓋率分數'] >= 80).sum(),
            '高忠誠度題數': (self.df['忠誠度分數'] >= 90).sum(),
            '優秀綜合評分題數': (self.df['綜合評分'] >= 85).sum()
        }
        return summary
    
    def save_results(self, output_path: str):
        """儲存評估結果"""
        # 建立時間戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_path}_評估結果_{timestamp}.xlsx"
        
        # 選擇要輸出的欄位
        output_columns = ['編號', '問題', '回答重點', 'UPGPT回答', 
                         '關鍵詞列表', '覆蓋率分數', '匹配關鍵詞',
                         '忠誠度分數', '忠誠度描述', '綜合評分']
        
        output_df = self.df[output_columns].copy()
        
        # 寫入Excel
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            output_df.to_excel(writer, sheet_name='評估結果', index=False)
            
            # 添加摘要頁
            summary_df = pd.DataFrame([self.generate_summary()]).T
            summary_df.columns = ['數值']
            summary_df.to_excel(writer, sheet_name='評估摘要')
            
            # 格式設定
            workbook = writer.book
            worksheet = writer.sheets['評估結果']
            
            # 設定列寬
            worksheet.set_column('A:A', 10)   # 編號
            worksheet.set_column('B:B', 40)   # 問題
            worksheet.set_column('C:D', 60)   # 回答內容
            worksheet.set_column('E:J', 20)   # 評分欄位
        
        print(f"✅ 評估結果已儲存至: {output_file}")
        return output_file

# 使用範例
if __name__ == "__main__":
    evaluator = RAGEvaluatorSingleModel("AI指導員-職災保護QA-測試題目.csv")
    results = evaluator.evaluate_all()
    summary = evaluator.generate_summary()
    
    print("\n📊 評估摘要:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # 儲存結果
    evaluator.save_results("職災保護QA")