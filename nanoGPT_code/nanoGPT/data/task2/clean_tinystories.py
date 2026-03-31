import re
import os

def clean_tiny_stories(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 使用 <|endoftext|> 作为分割符逻辑
    # 考虑到最后一个故事可能没有 EOT，我们先手动补一个或确保分割逻辑覆盖末尾
    stories = content.split('<|endoftext|>')

    cleaned_stories = []
    for story in stories:
        # 去除首尾空格/换行
        s = story.strip()
        if not s:
            continue
        
        # 2. 将故事内部的所有换行符（\n, \r\n）替换为空格
        # 这样能保证“每个故事一行”
        s = re.sub(r'\s+', ' ', s)
        
        cleaned_stories.append(s)

    # 3. 重新组装：每个故事间统一用两个 \n 隔离
    # 最终输出格式：Story1\n\nStory2\n\nStory3...
    final_output = '\n\n'.join(cleaned_stories)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_output)
    
    print(f"Successfully cleaned {len(cleaned_stories)} stories.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # 你可以根据实际文件名修改这两个路径
    input_file = "TinyStories-valid.txt" 
    output_file = "TinyStories_cleaned.txt"
    
    clean_tiny_stories(input_file, output_file)