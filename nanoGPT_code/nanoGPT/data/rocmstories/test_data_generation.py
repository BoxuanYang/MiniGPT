import tiktoken

src_path = './test.txt'
dest_path = './test_with_eot.txt'

# 显式使用 GPT-2 的 EOT 文本标记
EOT_TOKEN = "<|endoftext|>"

with open(src_path, 'r', encoding='utf-8') as f:
    stories = [line.strip() for line in f if line.strip()]

# 在每个故事末尾加上 EOT 标记，并用两个换行符分隔
# 这样 eval.py 读取时，每个段落末尾都会带上这个字符串
formatted_content = (EOT_TOKEN + '\n\n').join(stories) + EOT_TOKEN

with open(dest_path, 'w', encoding='utf-8') as f:
    f.write(formatted_content)

print(f"Done! 建议使用 test_with_eot.txt 进行评估。")