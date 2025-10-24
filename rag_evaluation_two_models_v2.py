"""
RAG 評估器 v2.0 - 三層評估架構
=================================

升級特性：
1. 第一層：關鍵詞匹配評估（快速）
2. 第二層：語義相似度評估（中等速度）
3. 第三層：GPT as a Judge 評估（深度分析，可選）
4. 靈活的權重配置系統
5. 完整的資料輸入輸出支援

作者：AI 知識庫優化團隊
版本：2.0
日期：2024
"""

import pandas as pd
import re
import jieba
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from datetime import datetime
import json
import os

# 第二層：語義相似度
try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️ 警告: sentence-transformers 未安裝，語義相似度功能將被停用")
    print("安裝方式: pip install sentence-transformers")

# 第三層：GPT 評審
try:
    import openai
    GPT_AVAILABLE = True
except ImportError:
    GPT_AVAILABLE = False
    print("⚠️ 警告: openai 未安裝，GPT 評審功能將被停用")
    print("安裝方式: pip install openai")


class RAGEvaluatorV2:
    """
    RAG 評估器 v2.0 - 三層評估架構

    評估層級：
    - Layer 1: 關鍵詞匹配（必選，快速）
    - Layer 2: 語義相似度（可選，推薦）
    - Layer 3: GPT Judge（可選，深度）
    """

    def __init__(
        self,
        excel_path: str,
        model_type: str = "cross",
        enable_semantic: bool = True,
        enable_gpt: bool = False,
        openai_api_key: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化 RAG 評估器 v2.0

        參數:
            excel_path: Excel 或 CSV 檔案路徑
            model_type: "vector", "smart_doc", 或 "cross"（跨技術比較）
            enable_semantic: 是否啟用語義相似度評估
            enable_gpt: 是否啟用 GPT 評審
            openai_api_key: OpenAI API 金鑰（啟用 GPT 評審時需要）
            weights: 評分權重配置 {"keyword": 0.3, "semantic": 0.3, "gpt": 0.4}
        """
        # 讀取資料
        if excel_path.lower().endswith('.csv'):
            self.df = pd.read_csv(excel_path, encoding='utf-8-sig')
        else:
            self.df = pd.read_excel(excel_path)

        self.model_type = model_type
        jieba.setLogLevel(20)

        # 評估層級配置
        self.enable_semantic = enable_semantic and SEMANTIC_AVAILABLE
        self.enable_gpt = enable_gpt and GPT_AVAILABLE

        # 初始化語義模型
        if self.enable_semantic:
            try:
                print("🔄 載入語義相似度模型...")
                # 使用 device='cpu' 避免 GPU 相關錯誤
                self.semantic_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    device='cpu'
                )
                print("✅ 語義相似度模型載入完成")
            except Exception as e:
                print(f"❌ 語義模型載入失敗: {str(e)}")
                print("⚠️ 將停用語義相似度評估")
                self.enable_semantic = False
                self.semantic_model = None
        else:
            self.semantic_model = None

        # 初始化 GPT 配置
        if self.enable_gpt:
            if openai_api_key:
                openai.api_key = openai_api_key
                print("✅ GPT 評審已啟用")
            else:
                print("⚠️ 警告: 未提供 OpenAI API 金鑰，GPT 評審將被停用")
                self.enable_gpt = False

        # 設定評分權重
        self.weights = self._initialize_weights(weights)

        # 自動偵測欄位
        self._detect_columns()

        print(f"\n📊 評估配置摘要:")
        print(f"  - 關鍵詞匹配: ✅ (權重: {self.weights['keyword']:.0%})")
        print(f"  - 語義相似度: {'✅' if self.enable_semantic else '❌'} (權重: {self.weights['semantic']:.0%})")
        print(f"  - GPT 評審: {'✅' if self.enable_gpt else '❌'} (權重: {self.weights['gpt']:.0%})")

    def _initialize_weights(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        """初始化評分權重"""
        if weights:
            return weights

        # 根據啟用的評估層級自動配置權重
        if self.enable_semantic and self.enable_gpt:
            return {"keyword": 0.3, "semantic": 0.3, "gpt": 0.4}
        elif self.enable_semantic:
            return {"keyword": 0.5, "semantic": 0.5, "gpt": 0.0}
        elif self.enable_gpt:
            return {"keyword": 0.4, "semantic": 0.0, "gpt": 0.6}
        else:
            return {"keyword": 1.0, "semantic": 0.0, "gpt": 0.0}

    def _detect_columns(self):
        """自動偵測資料欄位"""
        available_columns = self.df.columns.tolist()
        print(f"\n🔍 偵測到的欄位: {available_columns}")

        # 跨技術比較模式
        if self.model_type == "cross":
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
                raise ValueError(f"❌ 無法找到跨技術比較所需的欄位。可用欄位: {available_columns}")

        elif self.model_type == "vector":
            # 向量知識庫欄位偵測
            self.original_col = None
            self.optimized_col = None

            for col in available_columns:
                if '向量' in col and '原始' in col:
                    self.original_col = col
                elif '向量' in col and '彙整' in col:
                    self.optimized_col = col

            self.model_name = "向量知識庫"

            if not self.original_col or not self.optimized_col:
                raise ValueError(f"❌ 無法找到向量知識庫欄位。可用欄位: {available_columns}")

        else:  # smart_doc
            # 智慧文檔知識庫欄位偵測
            self.original_col = None
            self.optimized_col = None

            for col in available_columns:
                if ('智慧' in col or '文檔' in col) and '原始' in col:
                    self.original_col = col
                elif ('智慧' in col or '文檔' in col) and '彙整' in col:
                    self.optimized_col = col

            self.model_name = "智慧文檔知識庫"

            if not self.original_col or not self.optimized_col:
                raise ValueError(f"❌ 無法找到智慧文檔知識庫欄位。可用欄位: {available_columns}")

        print(f"✅ 使用欄位 - 原始: {self.original_col}, 優化: {self.optimized_col}")

    # ==================== 第一層：關鍵詞匹配評估 ====================

    def extract_keywords(self, text: str) -> List[str]:
        """從應回答詞彙中提取關鍵詞"""
        if pd.isna(text):
            return []

        # 移除編號和標點符號
        text = re.sub(r'\d+\.', '', text)
        text = re.sub(r'[：:。，,、\(\)]', ' ', text)

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

        # 使用 jieba 分詞處理剩餘文字
        words = jieba.cut(text)
        for word in words:
            if len(word.strip()) > 1 and word.strip() not in keywords:
                keywords.append(word.strip())

        return keywords

    def calculate_keyword_coverage(
        self,
        answer: str,
        keywords: List[str]
    ) -> Tuple[float, List[str], Dict]:
        """
        第一層評估：關鍵詞匹配覆蓋率

        返回:
            - coverage_score: 覆蓋率分數 (0-100)
            - matched_keywords: 匹配到的關鍵詞列表
            - details: 詳細分析數據
        """
        if pd.isna(answer) or not keywords:
            return 0.0, [], {"total": 0, "matched": 0, "missing": keywords}

        matched_keywords = []
        answer_lower = answer.lower()

        for keyword in keywords:
            if keyword.lower() in answer_lower:
                matched_keywords.append(keyword)
            elif self._is_similar_term(keyword, answer):
                matched_keywords.append(keyword)

        missing_keywords = [k for k in keywords if k not in matched_keywords]
        coverage_rate = len(matched_keywords) / len(keywords) if keywords else 0

        details = {
            "total_keywords": len(keywords),
            "matched_keywords": len(matched_keywords),
            "missing_keywords": len(missing_keywords),
            "missing_list": missing_keywords,
            "found_list": matched_keywords,
            "matched_ratio": coverage_rate
        }

        return coverage_rate * 100, matched_keywords, details

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

    # ==================== 第二層：語義相似度評估 ====================

    def calculate_semantic_similarity(
        self,
        reference_text: str,
        answer: str
    ) -> Tuple[float, Dict]:
        """
        第二層評估：語義相似度分析

        參數:
            reference_text: 應回答的詞彙/參考文本
            answer: 實際回答內容

        返回:
            - similarity_score: 語義相似度分數 (0-100)
            - details: 詳細分析數據
        """
        if not self.enable_semantic:
            return 0.0, {"error": "語義相似度功能未啟用"}

        if pd.isna(answer) or pd.isna(reference_text):
            return 0.0, {"error": "空白內容"}

        try:
            # 計算 embedding（不轉為 tensor，避免設備問題）
            embedding_ref = self.semantic_model.encode(
                reference_text,
                convert_to_tensor=False,
                device='cpu'
            )
            embedding_ans = self.semantic_model.encode(
                answer,
                convert_to_tensor=False,
                device='cpu'
            )

            # 使用 numpy 計算餘弦相似度
            import numpy as np
            from numpy.linalg import norm

            # 計算餘弦相似度
            similarity = np.dot(embedding_ref, embedding_ans) / (norm(embedding_ref) * norm(embedding_ans))

            # 轉換為 0-100 分數
            similarity_score = max(0, min(100, float(similarity) * 100))

            details = {
                "raw_similarity": float(similarity),
                "normalized_score": similarity_score,
                "reference_length": len(reference_text),
                "answer_length": len(answer)
            }

            return similarity_score, details

        except Exception as e:
            print(f"⚠️ 語義相似度計算錯誤: {str(e)}")
            return 0.0, {"error": str(e)}

    # ==================== 第三層：GPT as a Judge 評估 ====================

    def gpt_as_judge(
        self,
        question: str,
        reference_keywords: str,
        answer: str
    ) -> Tuple[Dict[str, float], str]:
        """
        第三層評估：GPT as a Judge 多維度評估

        參數:
            question: 測試問題
            reference_keywords: 應回答的詞彙
            answer: 實際回答內容

        返回:
            - scores: 多維度分數字典
            - reasoning: 評分理由
        """
        if not self.enable_gpt:
            return {
                "relevance": 0,
                "completeness": 0,
                "accuracy": 0,
                "faithfulness": 0,
                "overall": 0
            }, "GPT 評審功能未啟用"

        if pd.isna(answer):
            return {
                "relevance": 0,
                "completeness": 0,
                "accuracy": 0,
                "faithfulness": 0,
                "overall": 0
            }, "無回答內容"

        try:
            prompt = f"""你是一位專業的 RAG 系統評估專家。請評估以下回答的品質。

【問題】
{question}

【應包含的關鍵資訊】
{reference_keywords}

【實際回答】
{answer}

請從以下四個維度評分（0-100分）：

1. **相關性 (Relevance)**: 回答是否切題、是否回應了問題核心
2. **完整性 (Completeness)**: 是否包含了所有必要的關鍵資訊
3. **準確性 (Accuracy)**: 資訊是否正確、無明顯錯誤
4. **忠實度 (Faithfulness)**: 是否基於原始資料，無虛構或過度推測

請以 JSON 格式回傳評分結果：
{{
  "relevance": <0-100>,
  "completeness": <0-100>,
  "accuracy": <0-100>,
  "faithfulness": <0-100>,
  "overall": <0-100>,
  "reasoning": "簡短說明評分理由（2-3句話）"
}}

注意：overall 是四個維度的平均分數。"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            scores = {
                "relevance": result.get("relevance", 0),
                "completeness": result.get("completeness", 0),
                "accuracy": result.get("accuracy", 0),
                "faithfulness": result.get("faithfulness", 0),
                "overall": result.get("overall", 0)
            }

            reasoning = result.get("reasoning", "")

            return scores, reasoning

        except Exception as e:
            print(f"❌ GPT 評審錯誤: {str(e)}")
            return {
                "relevance": 0,
                "completeness": 0,
                "accuracy": 0,
                "faithfulness": 0,
                "overall": 0
            }, f"評估失敗: {str(e)}"

    # ==================== 綜合評估系統 ====================

    def evaluate_answer(
        self,
        question: str,
        reference_keywords: str,
        answer: str
    ) -> Dict:
        """
        綜合評估單個回答（三層評估）

        返回完整的評估結果字典
        """
        keywords = self.extract_keywords(reference_keywords)

        # 第一層：關鍵詞匹配
        keyword_score, matched, keyword_details = self.calculate_keyword_coverage(answer, keywords)

        # 第二層：語義相似度
        semantic_score, semantic_details = (0.0, {})
        if self.enable_semantic:
            semantic_score, semantic_details = self.calculate_semantic_similarity(
                reference_keywords, answer
            )

        # 第三層：GPT 評審
        gpt_scores, gpt_reasoning = ({}, "")
        if self.enable_gpt:
            gpt_scores, gpt_reasoning = self.gpt_as_judge(
                question, reference_keywords, answer
            )

        # 計算綜合評分
        final_score = (
            keyword_score * self.weights['keyword'] +
            semantic_score * self.weights['semantic'] +
            gpt_scores.get('overall', 0) * self.weights['gpt']
        )

        return {
            # 第一層結果
            "keyword_coverage": keyword_score,
            "matched_keywords": matched,
            "keyword_details": keyword_details,

            # 第二層結果
            "semantic_similarity": semantic_score,
            "semantic_details": semantic_details,

            # 第三層結果
            "gpt_scores": gpt_scores,
            "gpt_reasoning": gpt_reasoning,

            # 綜合結果
            "final_score": final_score,
            "weights_used": self.weights.copy()
        }

    def evaluate_all(self) -> pd.DataFrame:
        """執行完整評估 - 評估所有問題"""
        print(f"\n🚀 開始評估 {len(self.df)} 個問題...")

        # 初始化結果欄位
        result_columns = {
            # 原始版本
            'KEYWORD_COVERAGE_ORIGINAL': 0.0,
            'SEMANTIC_SIMILARITY_ORIGINAL': 0.0,
            'GPT_OVERALL_ORIGINAL': 0.0,
            'FINAL_SCORE_ORIGINAL': 0.0,
            'MATCHED_KEYWORDS_ORIGINAL': "",
            'GPT_REASONING_ORIGINAL': "",
            'ANSWER_ORIGINAL': self.df[self.original_col],

            # 優化版本
            'KEYWORD_COVERAGE_OPTIMIZED': 0.0,
            'SEMANTIC_SIMILARITY_OPTIMIZED': 0.0,
            'GPT_OVERALL_OPTIMIZED': 0.0,
            'FINAL_SCORE_OPTIMIZED': 0.0,
            'MATCHED_KEYWORDS_OPTIMIZED': "",
            'GPT_REASONING_OPTIMIZED': "",
            'ANSWER_OPTIMIZED': self.df[self.optimized_col]
        }

        for col, default_val in result_columns.items():
            if col not in ['ANSWER_ORIGINAL', 'ANSWER_OPTIMIZED']:
                self.df[col] = default_val
            else:
                self.df[col] = default_val

        # 逐行評估
        for idx, row in self.df.iterrows():
            if idx % 5 == 0:
                print(f"  進度: {idx + 1}/{len(self.df)} ({(idx + 1) / len(self.df) * 100:.1f}%)")

            question = row['測試問題']
            reference_keywords = row['應回答之詞彙']

            # 評估原始版本
            result_orig = self.evaluate_answer(
                question,
                reference_keywords,
                row[self.original_col]
            )

            self.df.at[idx, 'KEYWORD_COVERAGE_ORIGINAL'] = result_orig['keyword_coverage']
            self.df.at[idx, 'SEMANTIC_SIMILARITY_ORIGINAL'] = result_orig['semantic_similarity']
            self.df.at[idx, 'GPT_OVERALL_ORIGINAL'] = result_orig['gpt_scores'].get('overall', 0)
            self.df.at[idx, 'FINAL_SCORE_ORIGINAL'] = result_orig['final_score']
            self.df.at[idx, 'MATCHED_KEYWORDS_ORIGINAL'] = ', '.join(result_orig['matched_keywords'])
            self.df.at[idx, 'GPT_REASONING_ORIGINAL'] = result_orig['gpt_reasoning']

            # 評估優化版本
            result_opt = self.evaluate_answer(
                question,
                reference_keywords,
                row[self.optimized_col]
            )

            self.df.at[idx, 'KEYWORD_COVERAGE_OPTIMIZED'] = result_opt['keyword_coverage']
            self.df.at[idx, 'SEMANTIC_SIMILARITY_OPTIMIZED'] = result_opt['semantic_similarity']
            self.df.at[idx, 'GPT_OVERALL_OPTIMIZED'] = result_opt['gpt_scores'].get('overall', 0)
            self.df.at[idx, 'FINAL_SCORE_OPTIMIZED'] = result_opt['final_score']
            self.df.at[idx, 'MATCHED_KEYWORDS_OPTIMIZED'] = ', '.join(result_opt['matched_keywords'])
            self.df.at[idx, 'GPT_REASONING_OPTIMIZED'] = result_opt['gpt_reasoning']

        # 計算改善幅度
        self.df['KEYWORD_IMPROVEMENT'] = (
            self.df['KEYWORD_COVERAGE_OPTIMIZED'] - self.df['KEYWORD_COVERAGE_ORIGINAL']
        )
        self.df['SEMANTIC_IMPROVEMENT'] = (
            self.df['SEMANTIC_SIMILARITY_OPTIMIZED'] - self.df['SEMANTIC_SIMILARITY_ORIGINAL']
        )
        self.df['GPT_IMPROVEMENT'] = (
            self.df['GPT_OVERALL_OPTIMIZED'] - self.df['GPT_OVERALL_ORIGINAL']
        )
        self.df['FINAL_IMPROVEMENT'] = (
            self.df['FINAL_SCORE_OPTIMIZED'] - self.df['FINAL_SCORE_ORIGINAL']
        )

        print("✅ 評估完成！")
        return self.df

    def generate_summary_stats(self) -> Dict:
        """生成統計摘要"""
        stats = {
            '原始版本': {
                '平均關鍵詞覆蓋率': self.df['KEYWORD_COVERAGE_ORIGINAL'].mean(),
                '平均語義相似度': self.df['SEMANTIC_SIMILARITY_ORIGINAL'].mean(),
                '平均GPT評分': self.df['GPT_OVERALL_ORIGINAL'].mean(),
                '平均綜合評分': self.df['FINAL_SCORE_ORIGINAL'].mean(),
                '高分比例(≥80)': (self.df['FINAL_SCORE_ORIGINAL'] >= 80).sum() / len(self.df) * 100
            },
            '彙整優化版本': {
                '平均關鍵詞覆蓋率': self.df['KEYWORD_COVERAGE_OPTIMIZED'].mean(),
                '平均語義相似度': self.df['SEMANTIC_SIMILARITY_OPTIMIZED'].mean(),
                '平均GPT評分': self.df['GPT_OVERALL_OPTIMIZED'].mean(),
                '平均綜合評分': self.df['FINAL_SCORE_OPTIMIZED'].mean(),
                '高分比例(≥80)': (self.df['FINAL_SCORE_OPTIMIZED'] >= 80).sum() / len(self.df) * 100
            },
            '改善效果': {
                '平均關鍵詞覆蓋率提升': self.df['KEYWORD_IMPROVEMENT'].mean(),
                '平均語義相似度提升': self.df['SEMANTIC_IMPROVEMENT'].mean(),
                '平均GPT評分提升': self.df['GPT_IMPROVEMENT'].mean(),
                '平均綜合評分提升': self.df['FINAL_IMPROVEMENT'].mean(),
                '顯著改善比例(≥10)': (self.df['FINAL_IMPROVEMENT'] >= 10).sum() / len(self.df) * 100,
                '效果退步比例(<0)': (self.df['FINAL_IMPROVEMENT'] < 0).sum() / len(self.df) * 100
            },
            '評估配置': {
                '關鍵詞權重': self.weights['keyword'],
                '語義權重': self.weights['semantic'],
                'GPT權重': self.weights['gpt'],
                '語義相似度啟用': self.enable_semantic,
                'GPT評審啟用': self.enable_gpt
            }
        }

        return stats

    def save_results(self, output_path: str):
        """保存評估結果"""
        output_columns = ['序號', '測試資料', '測試問題', '應回答之詞彙']

        # 添加評分相關欄位
        output_columns.extend([
            'KEYWORD_COVERAGE_ORIGINAL', 'SEMANTIC_SIMILARITY_ORIGINAL',
            'GPT_OVERALL_ORIGINAL', 'FINAL_SCORE_ORIGINAL',
            'KEYWORD_COVERAGE_OPTIMIZED', 'SEMANTIC_SIMILARITY_OPTIMIZED',
            'GPT_OVERALL_OPTIMIZED', 'FINAL_SCORE_OPTIMIZED',
            'KEYWORD_IMPROVEMENT', 'SEMANTIC_IMPROVEMENT',
            'GPT_IMPROVEMENT', 'FINAL_IMPROVEMENT',
            'MATCHED_KEYWORDS_ORIGINAL', 'MATCHED_KEYWORDS_OPTIMIZED'
        ])

        # 如果啟用 GPT，添加推理欄位
        if self.enable_gpt:
            output_columns.extend(['GPT_REASONING_ORIGINAL', 'GPT_REASONING_OPTIMIZED'])

        output_df = self.df[[col for col in output_columns if col in self.df.columns]].copy()

        if output_path.lower().endswith('.csv'):
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, sheet_name='評分結果', index=False)

                workbook = writer.book
                worksheet = writer.sheets['評分結果']

                # 設定欄寬
                worksheet.set_column('A:A', 8)
                worksheet.set_column('B:B', 20)
                worksheet.set_column('C:C', 40)
                worksheet.set_column('D:D', 50)
                worksheet.set_column('E:P', 15)

        print(f"✅ 評分結果已保存到: {output_path}")


# 使用範例
if __name__ == "__main__":
    # 基本使用（僅關鍵詞匹配 + 語義相似度）
    evaluator = RAGEvaluatorV2(
        'test_data/AI指導員_測試腳本_v2拷貝.xlsx',
        model_type='cross',
        enable_semantic=True,
        enable_gpt=False
    )

    results = evaluator.evaluate_all()
    evaluator.save_results(f'評估結果_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')

    # 顯示統計
    print("\n📊 統計摘要:")
    stats = evaluator.generate_summary_stats()
    for category, metrics in stats.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
