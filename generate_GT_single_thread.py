import requests
import time
import os
from typing import Optional, List
import pandas as pd
import sys
import re

class LLMClient:
    def __init__(self, ip: str, port: str, model_name: str):
        self.ip = ip
        self.port = port
        self.model_name = model_name
        self.temperature = 0.8
        self.top_p = 0.95
        self.repetition_penalty = 1.1
        self.n = 10
        self.max_tokens = 4096
    
    
    def call_llm(self, prompt: str, max_retries: int = 3) -> Optional[List[str]]:
        """调用LLM API"""
        url = f"http://{self.ip}:{self.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": True},
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "n": self.n,
            "max_tokens": self.max_tokens
        }
        
        for attempt in range(max_retries):
            try:
                print(f"正在发送请求 (第 {attempt + 1} 次尝试)...")
                response = requests.post(url, headers=headers, json=payload, timeout=180) # 增加超时时间
                response.raise_for_status()
                data = response.json()
                choices = data.get('choices', [])
                print("成功获取响应。")
                return [choice['message']['content'] for choice in choices]
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"调用API失败（已重试{max_retries}次）: {str(e)}")
                    return None
                else:
                    print(f"第{attempt + 1}次调用失败，正在重试: {str(e)}")
                    time.sleep(5) # 增加重试等待时间
                    continue
            except Exception as e:
                print(f"未知错误: {str(e)}")
                return None
            

def read_prompt(file_path: str) -> str:
    """从文件读取prompt"""
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

def load_test_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """从 CSV 文件加载测试数据。"""
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None
    try:
        df = pd.read_csv(file_path).fillna('')
        print(f"成功从 {file_path} 加载了 {len(df)} 条测试数据。")
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return None

def construct_prompt(template: str, row: pd.Series) -> str:
    """根据DataFrame的一行数据和模板，构建prompt。"""
    prompt = template
    for col_name in ['question', 'env', 'hint', 'memory', 'dependency', 'state']:
        prompt = prompt.replace(f'{{{{ {col_name} }}}}', str(row.get(col_name, f"无[{col_name}]，请自行理解问题并给出答案")))
    return prompt

def extract_content_only(response: str) -> str:
    """
    从响应中提取答案内容（不包含think部分）
    
    Args:
        response (str): 模型响应内容
        
    Returns:
        str: 答案内容
    """
    # 使用正则表达式移除<think>...</think>标签及其内容
    # re.DOTALL 使得'.'可以匹配包括换行符在内的任意字符
    content_without_think = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    return content_without_think.strip()


if __name__ == "__main__":
    # --- 步骤 1: 初始化客户端和加载数据 ---
    main_client = LLMClient(
        ip="214.2.2.21",
        port="6091",
        model_name="qwen3_32b"
    )

    main_prompt_template = read_prompt('./data/prompt/bi_prompt_multi_template.txt')
    data_path = "./data/train_template.csv"
    test_dataframe = load_test_data_from_csv(data_path)
    output_path = "./output/output_data.csv"

    if test_dataframe is not None and main_prompt_template is not None:
        all_answers = [] # 用于存储所有行的答案

        for index, row in test_dataframe.iterrows():
            print(f"\n{'='*20} 正在处理第 {index + 1} / {len(test_dataframe)} 条数据 {'='*20}")
            
            # --- 步骤 2: 获取主模型输出 ---
            main_prompt = construct_prompt(main_prompt_template, row)
            
            # 一次调用获取多个答案
            model_outputs = main_client.call_llm(main_prompt)
            
            if model_outputs and len(model_outputs) == main_client.n:
                all_answers.append(model_outputs)
            else:
                print("未能获取足够的答案，将使用空字符串填充。")
                # 如果失败或答案数量不足，用 main_client.n 个空字符串填充
                all_answers.append([""] * main_client.n)
        # --- 步骤 3: 将答案添加到DataFrame并保存 ---
        print(f"\n{'='*20} 所有数据处理完毕，正在写入CSV文件... {'='*20}")
        
        # 创建新的列
        for i in range(main_client.n):
            col_name = f"ans{i+1}"
            test_dataframe[col_name] = [answers[i] for answers in all_answers]
            
        try:
            test_dataframe.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"结果已成功保存到 {output_path}")
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")

    else:
        print("程序因无法加载测试数据或Prompt模板而终止。")