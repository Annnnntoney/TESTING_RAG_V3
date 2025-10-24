"""
GPT 人工評審輔助工具
====================

功能：
1. 生成標準化的 GPT 評審 prompt
2. 支援批次生成（一次生成多題）
3. 支援單題深度分析
4. 自動格式化結果輸入欄位

使用方式：
1. 執行此腳本生成 prompts
2. 複製 prompt 到 ChatGPT
3. 將 ChatGPT 的回應貼回指定欄位
4. 系統自動解析並整合到評估結果

版本：2.0
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple


class GPTManualEvaluationHelper:
    """GPT 人工評審輔助工具"""

    def __init__(self, excel_path: str):
        """
        初始化輔助工具

        參數:
            excel_path: Excel 或 CSV 檔案路徑
        """
        if excel_path.lower().endswith('.csv'):
            self.df = pd.read_csv(excel_path, encoding='utf-8-sig')
        else:
            self.df = pd.read_excel(excel_path)

        print(f"✅ 已載入 {len(self.df)} 個問題")

    def generate_single_prompt(
        self,
        question_idx: int,
        version: str = "optimized"
    ) -> str:
        """
        生成單題評審 prompt

        參數:
            question_idx: 問題索引（從 0 開始）
            version: "original" 或 "optimized"

        返回:
            格式化的 prompt 文字
        """
        if question_idx >= len(self.df):
            return f"❌ 錯誤: 問題索引 {question_idx} 超出範圍（共 {len(self.df)} 題）"

        row = self.df.iloc[question_idx]

        # 自動偵測欄位名稱
        answer_col = self._detect_answer_column(version)

        if not answer_col:
            return f"❌ 錯誤: 無法找到 {version} 版本的回答欄位"

        question = row['測試問題']
        reference_keywords = row['應回答之詞彙']
        answer = row[answer_col]

        prompt = f"""你是一位專業的 RAG 系統評估專家。請評估以下回答的品質。

【問題 {row['序號']}】
{question}

【應包含的關鍵資訊】
{reference_keywords}

【實際回答（{version}版本）】
{answer}

請從以下四個維度評分（0-100分）：

1. **相關性 (Relevance)**: 回答是否切題、是否回應了問題核心
2. **完整性 (Completeness)**: 是否包含了所有必要的關鍵資訊
3. **準確性 (Accuracy)**: 資訊是否正確、無明顯錯誤
4. **忠實度 (Faithfulness)**: 是否基於原始資料，無虛構或過度推測

請以 JSON 格式回傳評分結果（請務必使用以下格式）：
{{
  "question_id": {row['序號']},
  "relevance": <0-100>,
  "completeness": <0-100>,
  "accuracy": <0-100>,
  "faithfulness": <0-100>,
  "overall": <0-100>,
  "reasoning": "簡短說明評分理由（2-3句話）",
  "strengths": ["優點1", "優點2"],
  "weaknesses": ["缺點1", "缺點2"]
}}

注意：
- overall 是四個維度的平均分數
- 請保持客觀公正
- 重點評估回答是否完整且忠實於原始資料
"""

        return prompt

    def generate_batch_prompts(
        self,
        start_idx: int = 0,
        end_idx: int = None,
        version: str = "optimized",
        questions_per_batch: int = 5
    ) -> List[str]:
        """
        生成批次評審 prompts

        參數:
            start_idx: 起始問題索引
            end_idx: 結束問題索引（None 表示到最後）
            version: "original" 或 "optimized"
            questions_per_batch: 每個 prompt 包含的問題數量

        返回:
            prompt 列表
        """
        if end_idx is None:
            end_idx = len(self.df)

        prompts = []

        for batch_start in range(start_idx, end_idx, questions_per_batch):
            batch_end = min(batch_start + questions_per_batch, end_idx)
            batch_prompts = []

            for idx in range(batch_start, batch_end):
                single_prompt = self.generate_single_prompt(idx, version)
                batch_prompts.append(single_prompt)

            # 將批次 prompts 合併
            combined_prompt = "\n\n" + "="*80 + "\n\n".join(batch_prompts)
            prompts.append(combined_prompt)

        return prompts

    def save_prompts_to_file(
        self,
        output_folder: str = "gpt_prompts",
        version: str = "optimized"
    ):
        """
        將所有 prompts 儲存到檔案

        參數:
            output_folder: 輸出資料夾路徑
            version: "original" 或 "optimized"
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 生成所有 prompts
        for idx in range(len(self.df)):
            prompt = self.generate_single_prompt(idx, version)

            # 儲存到檔案
            filename = f"{output_folder}/prompt_q{idx+1}_{version}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(prompt)

        print(f"✅ 已生成 {len(self.df)} 個 prompt 檔案到 {output_folder}/")

        # 生成批次檔案（5 題一組）
        batch_prompts = self.generate_batch_prompts(version=version, questions_per_batch=5)

        for batch_idx, batch_prompt in enumerate(batch_prompts):
            filename = f"{output_folder}/batch_prompt_{batch_idx+1}_{version}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(batch_prompt)

        print(f"✅ 已生成 {len(batch_prompts)} 個批次 prompt 檔案")

    def parse_gpt_response(self, response_text: str) -> Dict:
        """
        解析 ChatGPT 的 JSON 回應

        參數:
            response_text: ChatGPT 的回應文字

        返回:
            解析後的字典
        """
        try:
            # 嘗試直接解析 JSON
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            # 如果直接解析失敗，嘗試從文字中提取 JSON
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except:
                    pass

            return {
                "error": "無法解析 GPT 回應",
                "raw_response": response_text
            }

    def create_response_template(
        self,
        output_file: str = "gpt_responses_template.xlsx"
    ):
        """
        建立 GPT 回應輸入模板

        這個 Excel 檔案讓您可以：
        1. 看到每個問題的 prompt
        2. 貼上 ChatGPT 的 JSON 回應
        3. 系統自動解析並整合

        參數:
            output_file: 輸出檔案路徑
        """
        template_data = []

        for idx, row in self.df.iterrows():
            template_data.append({
                '序號': row['序號'],
                '測試問題': row['測試問題'],
                '應回答之詞彙': row['應回答之詞彙'],
                'Prompt_已生成': "請參考 gpt_prompts 資料夾",
                'ChatGPT回應_原始版本': "",  # 留空讓使用者填入
                'ChatGPT回應_優化版本': "",  # 留空讓使用者填入
                '狀態': "待評估"
            })

        template_df = pd.DataFrame(template_data)

        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            template_df.to_excel(writer, sheet_name='GPT評審回應', index=False)

            workbook = writer.book
            worksheet = writer.sheets['GPT評審回應']

            # 設定欄寬
            worksheet.set_column('A:A', 8)   # 序號
            worksheet.set_column('B:B', 40)  # 測試問題
            worksheet.set_column('C:C', 50)  # 應回答之詞彙
            worksheet.set_column('D:D', 30)  # Prompt
            worksheet.set_column('E:E', 80)  # ChatGPT回應_原始
            worksheet.set_column('F:F', 80)  # ChatGPT回應_優化
            worksheet.set_column('G:G', 15)  # 狀態

            # 設定標題格式
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4CAF50',
                'font_color': 'white',
                'border': 1
            })

            for col_num, value in enumerate(template_df.columns.values):
                worksheet.write(0, col_num, value, header_format)

        print(f"✅ 已建立 GPT 回應輸入模板: {output_file}")
        print(f"\n📝 使用步驟:")
        print(f"1. 打開 {output_file}")
        print(f"2. 從 gpt_prompts/ 資料夾複製 prompt 到 ChatGPT")
        print(f"3. 將 ChatGPT 的 JSON 回應貼到對應欄位")
        print(f"4. 執行 integrate_gpt_responses() 整合結果")

    def integrate_gpt_responses(
        self,
        response_file: str = "gpt_responses_template.xlsx",
        output_file: str = None
    ):
        """
        整合 GPT 評審回應到評估結果

        參數:
            response_file: 包含 GPT 回應的 Excel 檔案
            output_file: 輸出檔案路徑（None 則自動命名）
        """
        # 讀取回應檔案
        response_df = pd.read_excel(response_file)

        print(f"🔄 開始整合 GPT 評審結果...")

        # 初始化結果欄位
        self.df['GPT_RELEVANCE_ORIGINAL'] = 0
        self.df['GPT_COMPLETENESS_ORIGINAL'] = 0
        self.df['GPT_ACCURACY_ORIGINAL'] = 0
        self.df['GPT_FAITHFULNESS_ORIGINAL'] = 0
        self.df['GPT_OVERALL_ORIGINAL'] = 0
        self.df['GPT_REASONING_ORIGINAL'] = ""

        self.df['GPT_RELEVANCE_OPTIMIZED'] = 0
        self.df['GPT_COMPLETENESS_OPTIMIZED'] = 0
        self.df['GPT_ACCURACY_OPTIMIZED'] = 0
        self.df['GPT_FAITHFULNESS_OPTIMIZED'] = 0
        self.df['GPT_OVERALL_OPTIMIZED'] = 0
        self.df['GPT_REASONING_OPTIMIZED'] = ""

        success_count = 0
        error_count = 0

        # 解析每一行的 GPT 回應
        for idx, row in response_df.iterrows():
            # 解析原始版本回應
            if pd.notna(row['ChatGPT回應_原始版本']) and row['ChatGPT回應_原始版本'].strip():
                parsed = self.parse_gpt_response(row['ChatGPT回應_原始版本'])

                if 'error' not in parsed:
                    self.df.at[idx, 'GPT_RELEVANCE_ORIGINAL'] = parsed.get('relevance', 0)
                    self.df.at[idx, 'GPT_COMPLETENESS_ORIGINAL'] = parsed.get('completeness', 0)
                    self.df.at[idx, 'GPT_ACCURACY_ORIGINAL'] = parsed.get('accuracy', 0)
                    self.df.at[idx, 'GPT_FAITHFULNESS_ORIGINAL'] = parsed.get('faithfulness', 0)
                    self.df.at[idx, 'GPT_OVERALL_ORIGINAL'] = parsed.get('overall', 0)
                    self.df.at[idx, 'GPT_REASONING_ORIGINAL'] = parsed.get('reasoning', '')
                    success_count += 1
                else:
                    error_count += 1
                    print(f"⚠️ 問題 {row['序號']} 原始版本解析失敗")

            # 解析優化版本回應
            if pd.notna(row['ChatGPT回應_優化版本']) and row['ChatGPT回應_優化版本'].strip():
                parsed = self.parse_gpt_response(row['ChatGPT回應_優化版本'])

                if 'error' not in parsed:
                    self.df.at[idx, 'GPT_RELEVANCE_OPTIMIZED'] = parsed.get('relevance', 0)
                    self.df.at[idx, 'GPT_COMPLETENESS_OPTIMIZED'] = parsed.get('completeness', 0)
                    self.df.at[idx, 'GPT_ACCURACY_OPTIMIZED'] = parsed.get('accuracy', 0)
                    self.df.at[idx, 'GPT_FAITHFULNESS_OPTIMIZED'] = parsed.get('faithfulness', 0)
                    self.df.at[idx, 'GPT_OVERALL_OPTIMIZED'] = parsed.get('overall', 0)
                    self.df.at[idx, 'GPT_REASONING_OPTIMIZED'] = parsed.get('reasoning', '')
                    success_count += 1
                else:
                    error_count += 1
                    print(f"⚠️ 問題 {row['序號']} 優化版本解析失敗")

        # 計算改善幅度
        self.df['GPT_IMPROVEMENT'] = (
            self.df['GPT_OVERALL_OPTIMIZED'] - self.df['GPT_OVERALL_ORIGINAL']
        )

        # 儲存結果
        if output_file is None:
            output_file = f"評估結果_含GPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        self.df.to_excel(output_file, index=False)

        print(f"\n✅ 整合完成！")
        print(f"  - 成功解析: {success_count} 個回應")
        print(f"  - 解析失敗: {error_count} 個回應")
        print(f"  - 結果已儲存到: {output_file}")

        return self.df

    def _detect_answer_column(self, version: str) -> str:
        """自動偵測回答欄位名稱"""
        columns = self.df.columns.tolist()

        if version == "original":
            for col in columns:
                if '向量' in col and '原始' in col:
                    return col
                elif 'ANSWER_ORIGINAL' in col:
                    return col
        elif version == "optimized":
            for col in columns:
                if ('智慧' in col or '文檔' in col) and '彙整' in col:
                    return col
                elif 'ANSWER_OPTIMIZED' in col:
                    return col

        return None

    def generate_comparison_prompt(self, question_idx: int) -> str:
        """
        生成原始版本 vs 優化版本的對比評審 prompt

        這個 prompt 讓 ChatGPT 一次評估兩個版本並直接比較
        """
        if question_idx >= len(self.df):
            return f"❌ 錯誤: 問題索引 {question_idx} 超出範圍"

        row = self.df.iloc[question_idx]

        original_col = self._detect_answer_column("original")
        optimized_col = self._detect_answer_column("optimized")

        if not original_col or not optimized_col:
            return "❌ 錯誤: 無法找到回答欄位"

        prompt = f"""你是一位專業的 RAG 系統評估專家。請比較評估以下兩個版本的回答品質。

【問題 {row['序號']}】
{row['測試問題']}

【應包含的關鍵資訊】
{row['應回答之詞彙']}

【版本 A：原始版本】
{row[original_col]}

【版本 B：優化版本】
{row[optimized_col]}

請對兩個版本分別評分，並提供對比分析：

請以 JSON 格式回傳（請務必使用以下格式）：
{{
  "question_id": {row['序號']},
  "version_a": {{
    "relevance": <0-100>,
    "completeness": <0-100>,
    "accuracy": <0-100>,
    "faithfulness": <0-100>,
    "overall": <0-100>
  }},
  "version_b": {{
    "relevance": <0-100>,
    "completeness": <0-100>,
    "accuracy": <0-100>,
    "faithfulness": <0-100>,
    "overall": <0-100>
  }},
  "comparison": {{
    "improvement": <B的overall - A的overall>,
    "better_version": "A" or "B",
    "key_differences": ["差異1", "差異2", "差異3"],
    "recommendation": "優化建議"
  }},
  "reasoning": "整體評估說明（3-5句話）"
}}
"""

        return prompt


# 使用範例和命令行介面
if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("GPT 人工評審輔助工具 v2.0")
    print("=" * 80)

    # 檢查是否提供了檔案路徑
    if len(sys.argv) < 2:
        excel_file = input("\n請輸入 Excel 檔案路徑（或直接按 Enter 使用預設路徑）: ").strip()
        if not excel_file:
            excel_file = "test_data/AI指導員_測試腳本_v2拷貝.xlsx"
    else:
        excel_file = sys.argv[1]

    try:
        helper = GPTManualEvaluationHelper(excel_file)

        print("\n請選擇操作:")
        print("1. 生成所有 prompts 到檔案")
        print("2. 顯示單題 prompt（可複製到 ChatGPT）")
        print("3. 生成對比評審 prompt")
        print("4. 建立 GPT 回應輸入模板")
        print("5. 整合 GPT 回應到評估結果")

        choice = input("\n請輸入選項 (1-5): ").strip()

        if choice == "1":
            version = input("版本 (original/optimized, 預設 optimized): ").strip() or "optimized"
            helper.save_prompts_to_file(version=version)

        elif choice == "2":
            idx = int(input(f"問題編號 (1-{len(helper.df)}): ")) - 1
            version = input("版本 (original/optimized, 預設 optimized): ").strip() or "optimized"
            prompt = helper.generate_single_prompt(idx, version)
            print("\n" + "=" * 80)
            print("請複製以下內容到 ChatGPT:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        elif choice == "3":
            idx = int(input(f"問題編號 (1-{len(helper.df)}): ")) - 1
            prompt = helper.generate_comparison_prompt(idx)
            print("\n" + "=" * 80)
            print("請複製以下內容到 ChatGPT:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        elif choice == "4":
            helper.create_response_template()
            helper.save_prompts_to_file(version="original")
            helper.save_prompts_to_file(version="optimized")

        elif choice == "5":
            response_file = input("GPT 回應檔案路徑（預設 gpt_responses_template.xlsx）: ").strip()
            if not response_file:
                response_file = "gpt_responses_template.xlsx"

            result_df = helper.integrate_gpt_responses(response_file)
            print(f"\n✅ 整合完成！共處理 {len(result_df)} 個問題")

        else:
            print("無效的選項")

    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到檔案 {excel_file}")
    except Exception as e:
        print(f"❌ 錯誤: {str(e)}")
