import os
import random
import numpy as np
import tiktoken
import re

# --- 配置 ---
DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_SRC = os.path.join(DATA_DIR, "train.txt")
TEST_SRC  = os.path.join(DATA_DIR, "test.txt")

# 输出文件定义
OUT_TRAIN_TXT = os.path.join(DATA_DIR, "train_task2.txt")
OUT_TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")  # 建议加上 _task2 区分
OUT_VAL_TXT   = os.path.join(DATA_DIR, "val_task2.txt")
OUT_VAL_BIN   = os.path.join(DATA_DIR, "val.bin")
OUT_TEST_TXT  = os.path.join(DATA_DIR, "test_task2.txt")

NUM_TRAIN_TOTAL = 10000 
NUM_TEST_SAMPLES = 1000 
TRAIN_RATIO = 0.9

enc = tiktoken.get_encoding("gpt2")
EOT_ID = enc.eot_token

def clean_and_get_keywords(story, n=3):
    """
    清理标点并提取关键词。
    """
    words = re.findall(r'\b\w+\b', story)
    stop_words = {'the', 'with', 'they', 'their', 'there', 'about', 'would', 'could', 'from', 'this', 'that'}
    
    candidates = [w for w in words if w.lower() not in stop_words and len(w) > 3]
    candidates = sorted(list(set(candidates)), key=len, reverse=True)
    
    # 提取前n个关键词，用逗号分隔
    k_str = ", ".join(candidates[:n])
    # 按照要求，在关键词序列最后加一个句号
    if k_str:
        k_str += "."
    return k_str

def normalize_story(story):
    """清理故事内部的换行符，确保故事是一整块文本"""
    return " ".join(story.split())

def format_sample(line):
    story = normalize_story(line)
    if not story: return None
    keywords = clean_and_get_keywords(story)
    
    # 修改点：将原先的 \nStory 改为 Story，确保全在一个单行内
    # 现在的格式：Keywords: A, B, C. Story: Jack went...
    return f"Keywords: {keywords} Story: {story}"

def encode_content(formatted_text_list):
    """将格式化后的文本列表转为带EOT和\n\n分隔的token ids"""
    ids = []
    # 样本之间使用两个换行符作为逻辑分隔
    newline_ids = enc.encode("\n\n")
    for text in formatted_text_list:
        ids.extend(enc.encode(text, allowed_special={""})) # 显式处理可能存在的特殊字符
        ids.append(EOT_ID)
        ids.extend(newline_ids)
    return np.array(ids, dtype=np.uint16)

def main():
    random.seed(42)
    
    # --- 1. 处理训练和验证数据 ---
    if not os.path.exists(TRAIN_SRC):
        print(f"错误: 找不到输入文件 {TRAIN_SRC}")
        return

    with open(TRAIN_SRC, 'r', encoding='utf-8') as f:
        train_lines = [line.strip() for line in f if len(line.strip()) > 10]
    
    selected_train_raw = random.sample(train_lines, min(NUM_TRAIN_TOTAL, len(train_lines)))
    formatted_train_all = [format_sample(l) for l in selected_train_raw if format_sample(l)]
    
    # 划分训练/验证
    split_idx = int(len(formatted_train_all) * TRAIN_RATIO)
    train_samples = formatted_train_all[:split_idx]
    val_samples   = formatted_train_all[split_idx:]
    
    # 保存训练集 TXT 和 BIN
    with open(OUT_TRAIN_TXT, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(train_samples))
    encode_content(train_samples).tofile(OUT_TRAIN_BIN)
    
    # 保存验证集 TXT 和 BIN
    with open(OUT_VAL_TXT, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(val_samples))
    encode_content(val_samples).tofile(OUT_VAL_BIN)

    # --- 2. 处理测试数据 ---
    if os.path.exists(TEST_SRC):
        with open(TEST_SRC, 'r', encoding='utf-8') as f:
            test_lines = [line.strip() for line in f if len(line.strip()) > 10]
        
        selected_test = random.sample(test_lines, min(NUM_TEST_SAMPLES, len(test_lines)))
        formatted_test = [format_sample(l) for l in selected_test if format_sample(l)]
        
        with open(OUT_TEST_TXT, 'w', encoding='utf-8') as f:
            # 测试集同样使用 \n\n 分隔，确保 eval.py 正确按段落读取
            f.write("\n\n".join(formatted_test))
    else:
        print(f"警告: 找不到测试源文件 {TEST_SRC}，跳过测试集生成")

    print(f"Task 2 数据重构完成：")
    print(f" - 格式示例: Keywords: w1, w2, w3. Story: Once upon a time...")
    print(f" - 训练集: {len(train_samples)} 条 -> {OUT_TRAIN_BIN}")
    print(f" - 验证集: {len(val_samples)} 条 -> {OUT_VAL_BIN}")
    print(f" - 测试集: {len(formatted_test) if 'formatted_test' in locals() else 0} 条 -> {OUT_TEST_TXT}")

if __name__ == "__main__":
    main()