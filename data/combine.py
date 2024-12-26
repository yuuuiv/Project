import os
import pandas as pd
import glob

# 定义文件夹路径和文件名模式
folder_path = "data"  # 替换为包含 CSV 文件的文件夹路径
file_pattern = os.path.join(folder_path, "output_rf_*.csv")  # 匹配文件名为 output_rf_{}.csv 的文件
output_file = "combined_output_rf.csv"  # 合并后的文件名

# 获取所有符合模式的文件路径
csv_files = glob.glob(file_pattern)

# 检查是否找到符合条件的文件
if not csv_files:
    raise FileNotFoundError(f"未找到匹配的文件：{file_pattern}")

# 合并 CSV 文件
combined_df = pd.DataFrame()  # 创建一个空的 DataFrame

for file in csv_files:
    print(f"正在读取文件：{file}")
    df = pd.read_csv(file, parse_dates=["date"])  # 读取 CSV 文件，确保 date 列被解析为日期格式
    combined_df = pd.concat([combined_df, df], ignore_index=True)  # 将数据追加到 DataFrame 中

# 保存合并后的 DataFrame
output_path = os.path.join(folder_path, output_file)
combined_df.to_csv(output_path, index=False)
print(f"所有文件已合并，结果保存到：{output_path}")
