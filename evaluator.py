import requests
import time
import os
from typing import Optional, List, Dict, Tuple
import pandas as pd
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class LLMClient:
    def __init__(self, ip: str, port: str, model_name: str):
        self.ip = ip
        self.port = port
        self.model_name = model_name
        self.temperature = 0.8
        self.top_p = 0.95
        self.repetition_penalty = 1.1
        self.n = 5
        self.max_tokens = 4096
    
    
    def call_llm(self, prompt: str, session: Optional[requests.Session] = None, max_retries: int = 3) -> Optional[List[str]]:
        """调用LLM API (支持外部传入 Session 以复用连接)"""
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

        sess = session or requests.Session()
        
        for attempt in range(max_retries):
            try:
                print(f"[{threading.current_thread().name}] 正在发送请求 (第 {attempt + 1} 次尝试)...")
                response = sess.post(url, headers=headers, json=payload, timeout=180) # 增加超时时间
                response.raise_for_status()
                data = response.json()
                choices = data.get('choices', [])
                if not choices:
                    raise RuntimeError("响应中choices为空")
                return [choice['message']['content'] for choice in choices]
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"[{threading.current_thread().name}] 调用API失败（已重试{max_retries}次）: {str(e)}")
                    return None
                print(f"[{threading.current_thread().name}] 第{attempt + 1}次调用失败，正在重试: {str(e)}")
                time.sleep(5)
            except Exception as e:
                print(f"[{threading.current_thread().name}] 未知错误: {str(e)}")
                return None

def extract_content_only(response: str) -> str:
    # 使用正则表达式移除<think>...</think>标签及其内容，re.DOTALL 使得'.'可以匹配包括换行符在内的任意字符
    content_without_think = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    return content_without_think.strip()

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
    prompt = prompt.replace(f'{{{{ label }}}}', extract_content_only(str(row.get('label', f"无[label]，请自行理解问题并给出答案"))))
    return prompt

# 线程本地存储：每个线程复用自己的 requests.Session()
_thread_local = threading.local()

def _get_thread_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


def process_one_row(index: int, row: pd.Series, template: str, client: LLMClient) -> Tuple[int, List[str]]:
    """线程池任务：处理一行，返回 (index, answers)"""
    main_prompt = construct_prompt(template, row)
    session = _get_thread_session()
    model_outputs = client.call_llm(main_prompt, session=session)

    if model_outputs and len(model_outputs) == client.n:
        return index, [extract_content_only(output) for output in model_outputs]

    print(f"[{threading.current_thread().name}] 第 {index + 1} 条未拿到足够答案，使用空字符串填充。")
    return index, [""] * client.n

if __name__ == "__main__":
    # --- 步骤 1: 初始化客户端和加载数据 ---
    main_client = LLMClient(
        # ip="214.2.2.21",
        # port="6093",
        ip = "214.2.2.20",
        port = "6193",
        model_name="qwen3_32b",
    )

    main_prompt_template = read_prompt('./data/prompt/eval_prompt_v4.md')
    data_path = "./data/GRPO_train.csv"
    test_dataframe = load_test_data_from_csv(data_path)
    output_path = "./output/evaluation/GRPO_train_output.csv"

    if test_dataframe is None or main_prompt_template is None:
        print("程序因无法加载测试数据或Prompt模板而终止。")
        raise SystemExit(1)

    MAX_WORKERS = 6

    results: Dict[int, List[str]] = {}
    print(f"开始并发处理：共 {len(test_dataframe)} 条，max_workers={MAX_WORKERS}，每条请求返回 n={main_client.n} 个答案。")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row in test_dataframe.iterrows():
            futures.append(executor.submit(process_one_row, idx, row, main_prompt_template, main_client))

        completed = 0
        for fut in as_completed(futures):
            idx, answers = fut.result()
            results[idx] = answers
            completed += 1
            if completed % 10 == 0 or completed == len(test_dataframe):
                print(f"进度：{completed}/{len(test_dataframe)}")

    # 按原始顺序写回
    all_answers_ordered = [results[i] for i in range(len(test_dataframe))]

    for i in range(main_client.n):
        col_name = f"ans{i + 1}"
        test_dataframe[col_name] = [answers[i] for answers in all_answers_ordered]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        test_dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"结果已成功保存到 {output_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")


