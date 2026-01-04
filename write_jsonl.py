import os
import json
from typing import Optional, Dict, Any
import pandas as pd
import re


def read_prompt(file_path: str) -> Optional[str]:
    """从文件读取 prompt 模板"""
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

def extract_content_only(response: str) -> str:
    # 使用正则表达式移除<think>...</think>标签及其内容，re.DOTALL 使得'.'可以匹配包括换行符在内的任意字符
    content_without_think = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    return content_without_think.strip()

def construct_prompt(template: str, row: pd.Series) -> str:
    """根据 DataFrame 的一行数据和模板，构建 prompt。"""
    if template is None:
        raise ValueError("prompt 模板为空")

    prompt = template
    for col_name in ['question', 'env', 'hint', 'memory', 'dependency', 'state',]:
        prompt = prompt.replace(f'{{{{ {col_name} }}}}', str(row.get(col_name, '')))
    label_content = extract_content_only(str(row.get('label', '')))
    prompt = prompt.replace(f'{{{{ label }}}}', label_content)
    return prompt


def write_jsonl(file_path: str, data: Dict[str, Any], mode: str = "a") -> None:
    """将一条记录写入 JSONL 文件"""
    with open(file_path, mode, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def load_test_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """示例：从 CSV 读取 DataFrame，你可以用你自己的实现替换"""
    if not os.path.exists(file_path):
        print(f"错误：CSV 文件不存在 {file_path}")
        return None
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"读取 CSV 时出错: {str(e)}")
        return None


if __name__ == "__main__":
    bi_prompt_path = "./code/prompt/eval_prompt_v4.md"
    data_path = "./data/test&train/GRPO_train.csv"
    output_path = "./data/bi/GRPO_train.jsonl"

    bi_prompt = read_prompt(bi_prompt_path)
    test_dataframe = load_test_data_from_csv(data_path)

    if bi_prompt is None or test_dataframe is None:
        print("bi_prompt 或 test_dataframe 为空，程序终止。")
        exit(1)

    for _, row in test_dataframe.iterrows():
        user_prompt = construct_prompt(bi_prompt, row)
        record = {
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "GT_score": float(row.get("R_score", 0))
        }
        write_jsonl(output_path, record)
    print(f"已将数据写入 {output_path}")
