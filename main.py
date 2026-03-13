import pandas as pd
from pathlib import Path


Path("data").mkdir(parents=True, exist_ok=True)
Path("images").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)


url = "https://raw.githubusercontent.com/rgnhdyq/Sleep-Screen-Time-and-Stress-Analysis/main/data/sleep_mobile_stress_dataset_15000.csv"

df = pd.read_csv(url)

df.to_csv("data/Sleep_Health_and_Lifestyle_Dataset.csv", index=False)
print("数据集下载完成！")