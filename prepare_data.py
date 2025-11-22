# prepare_data.py
# Dataset Source: LUCAS Soil Survey 2009 (EU JRC), openly available at:
# https://esdac.jrc.ec.europa.eu/projects/lucas

import pandas as pd
import os

# 假设 LUCAS.SOIL_corr.csv 文件位于项目根目录。
RAW_FILE_NAME = "LUCAS.SOIL_corr.csv"
CLEANED_FILE_NAME = "cleaned_soil_data.csv"


def initial_clean_and_save():
    """
    功能：加载原始 LUCAS 数据，筛选目标列和光谱列，执行缺失值删除 (dropna)，
          并保存清理后的数据文件 (cleaned_soil_data.csv)。
    """
    # 检查原始文件是否存在
    if not os.path.exists(RAW_FILE_NAME):
        print(f"--- 错误：找不到原始数据文件 ---")
        print(f"请将 '{RAW_FILE_NAME}' 放在脚本所在的目录中。")
        return None

    print(f"-> 正在加载原始数据: {RAW_FILE_NAME}")
    try:
        df = pd.read_csv(RAW_FILE_NAME)
    except Exception as e:
        print(f"加载文件失败，请检查文件格式: {e}")
        return None

    # --- 2. 选择目标列和光谱数据列 ---
    target_columns = ['pH.in.CaCl2', 'OC', 'N']
    # 筛选以 'spc' 开头的列（假设光谱列均以此为前缀）
    spectra_columns = [col for col in df.columns if col.startswith('spc')]

    if not all(col in df.columns for col in target_columns):
        print(f"错误：原始文件缺少以下关键目标列: {target_columns}")
        return None

    # 合并需要保留的列
    cleaned_columns = spectra_columns + target_columns
    df_intermediate = df[cleaned_columns]

    # --- 3. 确保没有缺失值 (dropna) ---
    print("-> 正在执行缺失值删除 (dropna)...")
    # 删除所有目标列和光谱列中含有缺失值的行
    df_cleaned = df_intermediate.dropna(subset=cleaned_columns)

    print(f"   初始数据量: {len(df)}")
    print(f"   清洗后数据量: {len(df_cleaned)}")

    # --- 4. 保存清理后的数据 ---
    df_cleaned.to_csv(CLEANED_FILE_NAME, index=False)
    print(f"-> 清理后的数据已成功保存到: {CLEANED_FILE_NAME}")

    return df_cleaned


if __name__ == "__main__":
    initial_clean_and_save()