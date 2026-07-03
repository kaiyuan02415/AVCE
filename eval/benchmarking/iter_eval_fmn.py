#!/usr/bin/env python3
import os
import subprocess
import re
import argparse

# —— CLI 参数 —— 
parser = argparse.ArgumentParser(description="Batch run object_erase_s with optional vanilla mode")
parser.add_argument(
    '--vanilla', action='store_true',
    help='If set, pass --vanilla to object_erase_s to use the original SD model'
)
args = parser.parse_args()

# —— dataset ——
# c-50
DATASET_CSV_50 = "/home/kaiyuan/Code/DFM/dfm_mu_multi_re_v1/cp_get_all_apply_all_s/datasets/imagenet-mini-50.csv"
# c-15
DATASET_CSV_15 = "/home/kaiyuan/Code/DFM/dfm_mu_multi_re_v1/cp_get_all_apply_all_s/datasets/imagenet-simi-15.csv"
DATASET_CSV_15_10 = "/home/kaiyuan/Code/DFM/dfm_mu_multi_re_v1/cp_get_all_apply_all_s/datasets/imagenet-simi-15_10.csv"
# c-10
DATASET_CSV_10 = "/home/kaiyuan/Code/DFM/dfm_mu_multi_re_v1/cp_get_all_apply_all_s/datasets/imagenette.csv"

DATASET_CSV = DATASET_CSV_50

# —— checkpoint ——
# mus
CKPT_NAME_15 = "/home/kaiyuan/Code/DFM/dfm_mu_multi_v2/UCE/models/mus_p_2_am_0.85_en_20_c_15.pt"


CKPT_NAME = CKPT_NAME_15

# -- directory --
# mus
OUTPUT_SUMMARY_50 = "/home/kaiyuan/Code/DFM/dfm_mu_multi_re_v1/cp_get_all_apply_all_s/results/log_s/s_50/mus_p_2_am_0.7_en_20_c_15.txt"
\

OUTPUT_SUMMARY = OUTPUT_SUMMARY_50

# -- file --
LOG_FILE    = OUTPUT_SUMMARY.replace(".txt", ".log")
BASELINE    = "concept-prune"
REMOVAL     = "erase"
CUDA_DEVICE = "3"

# 要迭代的类别列表
CLASSES_C_50 = [
    "tabby", "Labrador retriever", "tiger", "lion", "African elephant",
    "sports car", "convertible", "school bus", "airliner",
    "mountain bike", "minivan", "pickup", "motor scooter",
    "folding chair", "rocking chair", "desk", "dining table", "table lamp",
    "acoustic guitar", "grand piano", "violin", "cornet", "sax",
    "cellular telephone", "reflex camera", "laptop", "television", "computer keyboard",
    "Granny Smith", "orange", "banana", "strawberry", "broccoli", "cauliflower",
    "cowboy hat", "running shoe", "sweatshirt", "jean", "trench coat",
    "pizza", "hotdog", "cheeseburger", "ice cream", "burrito", "mashed potato",
    "traffic light", "backpack", "umbrella", "bookcase", "water bottle",
]

CLASSES_C_10 = [
    "parachute",
    "golf ball",
    "garbage truck",
    "cassette player",
    "church",
    "tench",
    "english springer",
    "french horn",
    "chain saw",
    "gas pump",
]

CLASSES_C_15 = ["golden retriever", "labrador retriever", "german shepherd",
    "tabby", "tiger cat", "persian cat",
    "orange", "lemon", "pomegranate",
    "speedboat", "lifeboat", "yawl",
    "soccer ball", "volleyball", "tennis ball"
]

CLASSES_C_15_10 = [ "golden retriever","labrador retriever","german shepherd","Chesapeake Bay retriever","pug",
    "tabby","tiger cat","persian cat","Siamese cat","Egyptian cat",
    "orange","lemon","pomegranate","fig","Granny Smith",
    "speedboat","lifeboat","yawl","catamaran","schooner",
    "soccer ball","volleyball","tennis ball","rugby ball","ping-pong ball"
]

CLASSES = CLASSES_C_50

# modify done 

##################
# enter point
def main():
    # 确保输出目录存在，清空旧文件
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)
    open(OUTPUT_SUMMARY, 'w').close()
    open(LOG_FILE, 'w').close()

    results = {}
    pattern = re.compile(r"\[\d+\] 真实类别: (.+) \| 检测结果: (.+)")

    # 打开 log 文件
    with open(LOG_FILE, 'a') as logf:
        for cls in CLASSES:
            header = f"\n=== Processing: {cls} ===\n"
            print(header, end='')
            logf.write(header)
            logf.flush()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

            cmd = [
                "python", "benchmarking/object_erase_s.py",
                "--gpu", "0",
                "--target", cls,
                "--dataset_csv", DATASET_CSV,
                "--baseline", BASELINE,
                "--removal_mode", REMOVAL,
                "--ckpt_name", CKPT_NAME
            ]
            # 根据 --vanilla 开关添加参数
            if args.vanilla:
                cmd.append("--vanilla")

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env
            )

            correct = total = 0
            # 实时读取并输出
            for line in proc.stdout:
                print(line, end='')
                logf.write(line)
                logf.flush()
                m = pattern.search(line)
                if m:
                    label = m.group(1).strip().lower()
                    pred  = m.group(2).strip().lower()
                    total += 1
                    if pred == label:
                        correct += 1
            proc.wait()

            results[cls] = (correct, total)
            summary_line = f"{cls}: {correct}/{total}\n"
            print(summary_line, end='')
            logf.write(summary_line)
            logf.flush()

    # 写入 summary 文件
    with open(OUTPUT_SUMMARY, 'w') as sumf:
        for cls, (correct, total) in results.items():
            sumf.write(f"{cls}: {correct}/{total}\n")

    # 终端打印最终汇总
    print("\n=== Summary ===")
    for cls, (correct, total) in results.items():
        print(f"{cls}: {correct}/{total}")

if __name__ == "__main__":
    main()

