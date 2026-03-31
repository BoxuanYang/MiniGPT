import os

# ===== 配置 =====
INPUT_FILE  = "rocm_train.txt"
OUTPUT_FILE = "cleaned_rocm.txt"

def normalize_story(line: str) -> str:
    """
    清理单个故事：
    - 去掉首尾空格
    - 合并多余空白
    """
    return " ".join(line.strip().split())

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到输入文件: {INPUT_FILE}")
        return

    cleaned_stories = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            story = normalize_story(line)

            # 过滤掉空行或过短样本
            if len(story) > 10:
                cleaned_stories.append(story)

    print(f"✅ 读取到 {len(cleaned_stories)} 个 stories")

    # 用两个换行符连接
    output_text = "\n\n".join(cleaned_stories)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"✅ 已写入: {OUTPUT_FILE}")
    print(f"📌 格式: 每个 story 之间用 \\n\\n 分隔")

if __name__ == "__main__":
    main()