"""
評估歷史紀錄管理模組
===================
功能：
1. 儲存每次評估結果到 JSON
2. 載入歷史評估紀錄
3. 匯出完整報告到 Excel
4. 比較不同時間點的評估結果
"""

import json
import os
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional


LLM_JUDGE_TABLE_COLUMNS = [
    "timestamp",
    "excel_file",
    "question_id",
    "question",
    "reference_keywords",
    "answer",
    "version",
    "dimension",
    "score",
    "p",
    "q",
    "k",
    "r",
    "g",
    "shallow_flag",
    "positive_drivers",
    "negative_drivers",
    "on_topic_examples",
    "off_topic_examples",
    "covered",
    "partially",
    "missing",
    "correct_facts",
    "incorrect_facts",
    "unverifiable_facts",
    "essential",
    "supportive",
    "extraneous",
    "quality_notes",
    "coverage_debug",
    "k_debug",
    "reasoning",
    "raw_json"
]


class EvaluationHistoryManager:
    """評估歷史紀錄管理器"""

    def __init__(self, history_file: str = "evaluation_history.json"):
        """
        初始化管理器

        Args:
            history_file: 歷史紀錄檔案路徑
        """
        self.history_file = history_file
        self.history_data = self._load_history()
        self.judge_table_file = os.path.join(
            os.path.dirname(self.history_file) or '.',
            'llm_judge_table.csv'
        )

    def _load_history(self) -> Dict:
        """載入歷史紀錄"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 載入歷史紀錄失敗: {e}")
                return {"evaluations": []}
        else:
            return {"evaluations": []}

    def save_evaluation(
        self,
        excel_filename: str,
        question_id: int,
        question_text: str,
        reference_keywords: str,
        original_answer: str,
        optimized_answer: str,
        original_scores: Dict,
        optimized_scores: Dict,
        weights: Dict,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        儲存單次評估結果

        Args:
            excel_filename: Excel 檔案名稱
            question_id: 問題編號
            question_text: 問題內容
            reference_keywords: 參考關鍵字
            original_answer: 原始回答
            optimized_answer: 優化回答
            original_scores: 原始版本評分
            optimized_scores: 優化版本評分
            weights: 評分權重
            metadata: 額外元資料

        Returns:
            是否成功儲存
        """
        try:
            # 建立評估紀錄
            evaluation_record = {
                "timestamp": datetime.now().isoformat(),
                "excel_file": excel_filename,
                "question_id": question_id,
                "question": question_text,
                "reference_keywords": reference_keywords,
                "answers": {
                    "original": original_answer,
                    "optimized": optimized_answer
                },
                "scores": {
                    "original": original_scores,
                    "optimized": optimized_scores
                },
                "weights": weights,
                "metadata": metadata or {}
            }

            # 加入歷史紀錄
            self.history_data["evaluations"].append(evaluation_record)

            # 儲存到檔案
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history_data, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            print(f"❌ 儲存評估紀錄失敗: {e}")
            return False

    def append_llm_judge_records(self, records: List[Dict]) -> bool:
        """將 LLM-as-Judge 分項評分追加到表格檔。"""
        if not records:
            return False

        try:
            df = pd.DataFrame(records)
            df = df.reindex(columns=LLM_JUDGE_TABLE_COLUMNS)
            df = df.where(pd.notnull(df), '')
            file_exists = os.path.exists(self.judge_table_file)
            df.to_csv(
                self.judge_table_file,
                mode='a',
                header=not file_exists,
                index=False
            )
            return True
        except Exception as e:
            print(f"❌ 儲存 LLM-as-Judge 表格失敗: {e}")
            return False

    def load_llm_judge_table(self) -> pd.DataFrame:
        """載入 LLM-as-Judge 表格資料。"""
        if os.path.exists(self.judge_table_file):
            try:
                return pd.read_csv(self.judge_table_file)
            except Exception as e:
                print(f"⚠️ 載入 LLM-as-Judge 表格失敗: {e}")
        return pd.DataFrame(columns=LLM_JUDGE_TABLE_COLUMNS)

    def get_all_evaluations(self) -> List[Dict]:
        """取得所有評估紀錄"""
        return self.history_data.get("evaluations", [])

    def get_evaluations_by_file(self, excel_filename: str) -> List[Dict]:
        """取得特定檔案的所有評估紀錄"""
        return [
            eval_record for eval_record in self.history_data.get("evaluations", [])
            if eval_record.get("excel_file") == excel_filename
        ]

    def get_evaluations_by_date(self, start_date: str, end_date: str) -> List[Dict]:
        """取得特定日期範圍的評估紀錄"""
        evaluations = []
        for eval_record in self.history_data.get("evaluations", []):
            timestamp = eval_record.get("timestamp", "")
            if start_date <= timestamp <= end_date:
                evaluations.append(eval_record)
        return evaluations

    def export_to_excel(self, output_path: str, evaluations: Optional[List[Dict]] = None) -> bool:
        """
        匯出評估紀錄到 Excel

        Args:
            output_path: 輸出檔案路徑
            evaluations: 要匯出的評估紀錄（None = 全部）

        Returns:
            是否成功匯出
        """
        try:
            if evaluations is None:
                evaluations = self.get_all_evaluations()

            if not evaluations:
                print("⚠️ 沒有評估紀錄可匯出")
                return False

            # 建立 DataFrame
            rows = []
            for eval_record in evaluations:
                # 基本資訊
                row = {
                    "評估時間": eval_record.get("timestamp", ""),
                    "檔案名稱": eval_record.get("excel_file", ""),
                    "問題編號": eval_record.get("question_id", ""),
                    "問題": eval_record.get("question", ""),
                    "參考關鍵字": eval_record.get("reference_keywords", ""),
                }

                # 原始版本
                original_scores = eval_record.get("scores", {}).get("original", {})
                row.update({
                    "原始-關鍵詞": original_scores.get("keyword_score", ""),
                    "原始-語義": original_scores.get("semantic_score", ""),
                    "原始-GPT相關性": original_scores.get("gpt_relevance", ""),
                    "原始-GPT完整性": original_scores.get("gpt_completeness", ""),
                    "原始-GPT準確性": original_scores.get("gpt_accuracy", ""),
                    "原始-GPT忠實度": original_scores.get("gpt_faithfulness", ""),
                    "原始-GPT總分": original_scores.get("gpt_overall", ""),
                    "原始-綜合評分": original_scores.get("final_score", ""),
                    "原始-回答": eval_record.get("answers", {}).get("original", ""),
                })

                # 優化版本
                optimized_scores = eval_record.get("scores", {}).get("optimized", {})
                row.update({
                    "優化-關鍵詞": optimized_scores.get("keyword_score", ""),
                    "優化-語義": optimized_scores.get("semantic_score", ""),
                    "優化-GPT相關性": optimized_scores.get("gpt_relevance", ""),
                    "優化-GPT完整性": optimized_scores.get("gpt_completeness", ""),
                    "優化-GPT準確性": optimized_scores.get("gpt_accuracy", ""),
                    "優化-GPT忠實度": optimized_scores.get("gpt_faithfulness", ""),
                    "優化-GPT總分": optimized_scores.get("gpt_overall", ""),
                    "優化-綜合評分": optimized_scores.get("final_score", ""),
                    "優化-回答": eval_record.get("answers", {}).get("optimized", ""),
                })

                # 評分改善
                row["綜合評分改善"] = optimized_scores.get("final_score", 0) - original_scores.get("final_score", 0)

                # 權重設定
                weights = eval_record.get("weights", {})
                row.update({
                    "權重-關鍵詞": weights.get("keyword", ""),
                    "權重-語義": weights.get("semantic", ""),
                    "權重-GPT": weights.get("gpt", ""),
                })

                rows.append(row)

            df = pd.DataFrame(rows)

            # 匯出到 Excel
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"✅ 成功匯出 {len(rows)} 筆評估紀錄到 {output_path}")
            return True

        except Exception as e:
            print(f"❌ 匯出 Excel 失敗: {e}")
            return False

    def get_statistics(self) -> Dict:
        """取得評估統計資訊"""
        evaluations = self.get_all_evaluations()

        if not evaluations:
            return {
                "total_evaluations": 0,
                "files_evaluated": 0,
                "avg_improvement": 0,
                "questions_with_improvement": 0
            }

        # 計算統計
        files = set(eval_record.get("excel_file") for eval_record in evaluations)
        improvements = []

        for eval_record in evaluations:
            original_score = eval_record.get("scores", {}).get("original", {}).get("final_score", 0)
            optimized_score = eval_record.get("scores", {}).get("optimized", {}).get("final_score", 0)
            improvement = optimized_score - original_score
            improvements.append(improvement)

        return {
            "total_evaluations": len(evaluations),
            "files_evaluated": len(files),
            "avg_improvement": sum(improvements) / len(improvements) if improvements else 0,
            "questions_with_improvement": sum(1 for imp in improvements if imp > 0),
            "questions_with_decline": sum(1 for imp in improvements if imp < 0),
            "questions_unchanged": sum(1 for imp in improvements if imp == 0)
        }

    def clear_history(self) -> bool:
        """清除所有歷史紀錄"""
        try:
            self.history_data = {"evaluations": []}
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ 清除歷史紀錄失敗: {e}")
            return False
