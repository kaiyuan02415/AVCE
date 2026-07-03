from datasets import load_dataset
from collections import Counter

# 加载 I2P 数据集
dataset = load_dataset("AIML-TUDA/i2p")["train"]

cat_counter = Counter()

for entry in dataset:
    # 假设 categories 是以逗号分隔的字符串
    cats = entry["categories"].split(",")
    for cat in cats:
        cat = cat.strip()
        if cat:  # 忽略空字符串
            cat_counter[cat] += 1

# 输出所有类别及其出现次数
for category, count in cat_counter.most_common():
    print(f"{category}: {count}")
