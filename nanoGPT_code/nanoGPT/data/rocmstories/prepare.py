import os
import numpy as np
import tiktoken

# 获取当前路径
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SRC = os.path.join(DATA_DIR, "train.txt")
TEST_SRC  = os.path.join(DATA_DIR, "test.txt")

def main():
    # 1. 强制使用 GPT-2 分词器
    enc = tiktoken.get_encoding("gpt2")

    # 2. 直接读取最原始的 test.txt 和 train.txt（不改任何一个字）
    print("Reading files...")
    with open(TRAIN_SRC, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(TEST_SRC, 'r', encoding='utf-8') as f:
        test_text = f.read()

    # 3. 编码普通文本（不加任何特殊 Token）
    print("Encoding...")
    train_ids = enc.encode_ordinary(train_text)
    test_ids = enc.encode_ordinary(test_text)

    # 4. 按 9:1 划分 Train 和 Val
    split_idx = int(len(train_ids) * 0.9)
    train_data = train_ids[:split_idx]
    val_data = train_ids[split_idx:]

    # 5. 保存为 bin 文件
    print("Writing .bin files...")
    np.array(train_data, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "train.bin"))
    np.array(val_data, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "val.bin"))
    np.array(test_ids, dtype=np.uint16).tofile(os.path.join(DATA_DIR, "test.bin"))

    # 6. [关键保命操作] 自动删除 meta.pkl，防止 eval.py 崩溃
    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    if os.path.exists(meta_path):
        os.remove(meta_path)
        print("Deleted old meta.pkl to ensure eval.py works perfectly.")

    print("Done! Ready to train.")

if __name__ == "__main__":
    main()