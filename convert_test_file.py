import pandas as pd
import sys

def convert_test_file(input_file, output_file):
    """將測試檔案轉換為系統相容格式"""
    # 讀取原始檔案
    df_original = pd.read_csv(input_file)
    
    # 建立新的DataFrame，符合系統格式
    df_new = pd.DataFrame()
    
    # 對應欄位
    df_new['序號'] = df_original['編號']
    df_new['測試資料'] = '職災保護QA'  # 固定值或可自訂
    df_new['測試問題'] = df_original['問題']
    df_new['應回答之詞彙'] = df_original['回答重點']
    
    # 將UPGPT回答複製到四個評估方法欄位
    # 方案A: 使用同一個回答測試所有方法
    df_new['向量知識庫（原始版）'] = df_original['UPGPT回答']
    df_new['向量知識庫（彙整版）'] = df_original['UPGPT回答']
    df_new['智慧文檔知識庫（原始版）'] = df_original['UPGPT回答']
    df_new['智慧文檔知識庫（彙整版）'] = df_original['UPGPT回答']
    
    # 儲存為Excel格式
    df_new.to_excel(output_file, index=False)
    print(f"✅ 轉換完成！檔案已儲存至: {output_file}")
    print(f"   - 原始資料筆數: {len(df_original)}")
    print(f"   - 轉換後資料筆數: {len(df_new)}")

# 執行轉換
if __name__ == "__main__":
    input_file = "AI指導員-職災保護QA-測試題目.csv"
    output_file = "test_data/職災保護QA_轉換格式.xlsx"
    
    try:
        convert_test_file(input_file, output_file)
    except Exception as e:
        print(f"❌ 轉換失敗: {str(e)}")