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
        self.temperature = 0.0
        self.top_p = 0.95
        self.repetition_penalty = 1.1
        self.n = 1
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

def construct_prompt(template: str, row: pd.Series, candidate_text: Optional[str] = None) -> str:
    """根据DataFrame的一行数据和模板，构建prompt。candidate_text 用于覆盖 {{ label }} 的填充内容。"""
    prompt = template
    for col_name in ['question', 'env', 'hint', 'memory', 'dependency', 'state']:
        prompt = prompt.replace(
            f'{{{{ {col_name} }}}}',
            str(row.get(col_name, f"无[{col_name}]，请自行理解问题并给出答案"))
        )

    # 关键：{{ label }} 作为“待评估答案”的占位符
    if candidate_text is None:
        candidate_text = str(row.get('label', f"无[label]，请自行理解问题并给出答案"))
    prompt = prompt.replace('{{ label }}', extract_content_only(str(candidate_text)))
    return prompt


# 线程本地存储：每个线程复用自己的 requests.Session()
_thread_local = threading.local()

def _get_thread_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


def process_one_row(index: int, row: pd.Series, template: str, client: LLMClient) -> Tuple[int, Dict[str, str]]:
    """
    线程池任务：对 label/ans1..ans5 分别评估。
    返回 (index, results_dict)，results_dict 里是各列评估输出。
    """
    session = _get_thread_session()

    fields = ["label", "ans1", "ans2", "ans3", "ans4", "ans5"]
    out: Dict[str, str] = {}

    for f in fields:
        candidate = row.get(f, "")
        prompt = construct_prompt(template, row, candidate_text=candidate)
        model_outputs = client.call_llm(prompt, session=session)

        # 评估一般 n=1；若你仍然保留 n>1，这里取第1个，也可改为拼接
        if model_outputs and len(model_outputs) >= 1:
            out[f"eval_{f}"] = extract_content_only(model_outputs[0])
        else:
            print(f"[{threading.current_thread().name}] 第 {index + 1} 条的 {f} 评估失败，填充空字符串。")
            out[f"eval_{f}"] = ""

    return index, out


if __name__ == "__main__":
    # --- 步骤 1: 初始化客户端和加载数据 ---
    main_client = LLMClient(
        ip="214.2.2.20",
        port="6191",
        model_name="qwen3_32b",
    )

    main_prompt_template = read_prompt('./data/prompt/eval_prompt_v4.md')
    data_path = "./data/test&train/test1231.csv"
    test_dataframe = load_test_data_from_csv(data_path)

    output_path = "./output/evaluation/test1231_eval.csv"

    if test_dataframe is None or main_prompt_template is None:
        print("程序因无法加载测试数据或Prompt模板而终止。")
        raise SystemExit(1)

    # 可选：检查必要列
    required_cols = ['question', 'env', 'hint', 'memory', 'dependency', 'state', 'label', 'ans1', 'ans2', 'ans3', 'ans4', 'ans5']
    missing = [c for c in required_cols if c not in test_dataframe.columns]
    if missing:
        print(f"错误：CSV缺少必要列：{missing}")
        raise SystemExit(1)

    MAX_WORKERS = 6

    results: Dict[int, Dict[str, str]] = {}
    print(f"开始并发评估：共 {len(test_dataframe)} 条，max_workers={MAX_WORKERS}，每条评估 6 次。")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row in test_dataframe.iterrows():
            futures.append(executor.submit(process_one_row, idx, row, main_prompt_template, main_client))

        completed = 0
        for fut in as_completed(futures):
            idx, eval_dict = fut.result()
            results[idx] = eval_dict
            completed += 1
            if completed % 10 == 0 or completed == len(test_dataframe):
                print(f"进度：{completed}/{len(test_dataframe)}")

    # 按原顺序写回新增评估列
    ordered = [results[i] for i in range(len(test_dataframe))]

    eval_cols = ["eval_label", "eval_ans1", "eval_ans2", "eval_ans3", "eval_ans4", "eval_ans5"]
    for col in eval_cols:
        test_dataframe[col] = [ordered[i].get(col, "") for i in range(len(test_dataframe))]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        test_dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"结果已成功保存到 {output_path}")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")



