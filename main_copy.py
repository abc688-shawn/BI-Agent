import os
import re
import json
import time
import math
import requests
from typing import Optional, List, Dict, Any
import pandas as pd
import tiktoken
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from swanlab.plugin.notification import LarkCallback
from openai import OpenAI


# =========================
# 配置区
# =========================

# 飞书机器人配置 - 自动化评估bot
lark_callback = LarkCallback(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/986352c7-a346-4e33-9126-c7fd9208b5a0",
    secret="lwSomSfbR4XZpX33cIhWYb",
)

# 本地 tokenizer（用于 Qwen token 计数）
qwen_tokenizer = AutoTokenizer.from_pretrained(
    "/data01/lanjunfei/code/output/agent/Qwen3_14B/swift_image/1124/2_dense_agent_v24_liantiao_think/hf_output"
)

# tiktoken encodings
try:
    gpt4_encoding = tiktoken.encoding_for_model("gpt-4.1")
except Exception:
    gpt4_encoding = tiktoken.get_encoding("cl100k_base")

try:
    deepseek_reasoner_encoding = tiktoken.get_encoding("deepseek-reasoner")
except Exception:
    deepseek_reasoner_encoding = tiktoken.get_encoding("cl100k_base")

try:
    deepseek_chat_encoding = tiktoken.get_encoding("deepseek-chat")
except Exception:
    deepseek_chat_encoding = tiktoken.get_encoding("cl100k_base")


# =========================
# Clients
# =========================

class LLMClient:
    def __init__(self, ip: str, port: str, model_name: str):
        self.ip = ip
        self.port = port
        self.model_name = model_name
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.1
        self.n = 1

    def call_llm(
        self,
        prompt: str,
        max_retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
    ) -> Optional[List[str]]:
        """调用本地 LLM API，支持覆盖 temperature / top_p / n"""
        url = f"http://{self.ip}:{self.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        nn = n if n is not None else self.n

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": True},
            "temperature": temp,
            "top_p": tp,
            "repetition_penalty": self.repetition_penalty,
            "n": nn,
            "max_tokens": 4096,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                return [choice["message"]["content"] for choice in choices]

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"调用API失败（已重试{max_retries}次）: {str(e)}")
                    return None
                time.sleep(2)

            except Exception as e:
                print(f"未知错误: {str(e)}")
                return None

        return None

    def count_tokens(self, text: str) -> int:
        """使用本地 Qwen tokenizer 统计 token 数量"""
        if not text:
            return 0
        input_ids = qwen_tokenizer.encode(text)
        return len(input_ids)


class OpenAIClient:
    """使用 OpenAI ChatGPT 的客户端，接口仿照 LLMClient.call_llm"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.temperature = 0.7
        self.top_p = 0.9
        self.n = 1

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def call_llm(
        self,
        prompt: str,
        max_retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
    ) -> Optional[List[str]]:

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("错误：未找到 OPENAI_API_KEY 环境变量，请先配置。")
            return None

        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        nn = n if n is not None else self.n

        for attempt in range(max_retries):
            try:
                client = OpenAI(api_key=api_key)
                chat_completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    top_p=tp,
                    n=nn,
                    max_tokens=4096,
                )
                return [choice.message.content for choice in chat_completion.choices]

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"调用 OpenAI API 失败（已重试 {max_retries} 次）: {str(e)}")
                    return None
                time.sleep(2)

        return None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))


class DeepSeekClient:
    """使用 DeepSeek 作为主模型，接口仿照 LLMClient / OpenAIClient"""

    def __init__(self, model_name: str = "deepseek-chat"):
        self.model_name = model_name
        self.temperature = 0.7
        self.top_p = 0.9
        self.n = 1

        lower_name = (model_name or "").lower()
        if "reasoner" in lower_name:
            self.encoding = deepseek_reasoner_encoding
        else:
            self.encoding = deepseek_chat_encoding

    def call_llm(
        self,
        prompt: str,
        max_retries: int = 3,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
    ) -> Optional[List[str]]:

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("错误：未找到 DEEPSEEK_API_KEY 环境变量，请先配置。")
            return None

        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p
        nn = n if n is not None else self.n

        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )
                chat_completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    top_p=tp,
                    n=nn,
                    max_tokens=4096,
                )
                outputs: List[str] = []

                is_reasoner = "reasoner" in (self.model_name or "").lower()
                for choice in chat_completion.choices:
                    msg = choice.message
                    if is_reasoner:
                        reasoning = ""
                        if hasattr(msg, "model_extra") and isinstance(msg.model_extra, dict):
                            reasoning = msg.model_extra.get("reasoning_content", "") or ""
                        final_content = msg.content or ""
                        content = f"<think>\n{reasoning}\n</think>\n{final_content}"
                    else:
                        content = msg.content or ""
                    outputs.append(content)

                return outputs

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"调用 DeepSeek API 失败（已重试 {max_retries} 次）: {str(e)}")
                    return None
                time.sleep(2)

        return None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))


# =========================
# Utils
# =========================

def validate_structure_integrity(model_output: str) -> Dict[str, Any]:
    """使用正则表达式验证模型输出的结构完整性"""
    task_completed_pattern = r"当前任务已完成，无需工具调用|任务已完成|无需工具调用"
    is_task_completed = re.search(task_completed_pattern, model_output) is not None

    if is_task_completed:
        think_count = len(re.findall(r"<think>", model_output))
        think_close_count = len(re.findall(r"</think>", model_output))
        plan_count = len(re.findall(r"<plan>", model_output))
        python_count = len(re.findall(r"```python", model_output))

        if think_count == 1 and think_close_count == 1 and plan_count == 0 and python_count == 0:
            return {"reason": "任务已完成的特例：正确包含一次<think>...</think>和完成说明，无其他结构。", "score": 1}
        return {
            "reason": f"任务已完成的特例格式错误：<think>出现{think_count}次，</think>出现{think_close_count}次，<plan>出现{plan_count}次，```python出现{python_count}次。",
            "score": 0,
        }

    think_matches = re.findall(r"<think>.*?</think>", model_output, re.DOTALL)
    plan_matches = re.findall(r"<plan>.*?</plan>", model_output, re.DOTALL)
    python_matches = re.findall(r"```python.*?```", model_output, re.DOTALL)

    think_open_count = len(re.findall(r"<think>", model_output))
    think_close_count = len(re.findall(r"</think>", model_output))
    plan_open_count = len(re.findall(r"<plan>", model_output))
    plan_close_count = len(re.findall(r"</plan>", model_output))

    other_tags = re.findall(r"<(?!think|/think|plan|/plan)(\w+)>", model_output)

    if len(think_matches) != 1:
        return {
            "reason": f"<think>...</think>应出现1次，实际出现{len(think_matches)}次（开标签{think_open_count}次，闭标签{think_close_count}次）。",
            "score": 0,
        }

    if len(plan_matches) != 1:
        return {
            "reason": f"<plan>...</plan>应出现1次，实际出现{len(plan_matches)}次（开标签{plan_open_count}次，闭标签{plan_close_count}次）。",
            "score": 0,
        }

    if len(python_matches) != 1:
        return {"reason": f"```python...```代码块应出现1次，实际出现{len(python_matches)}次。", "score": 0}

    if other_tags:
        return {"reason": f"输出包含不允许的标签：{', '.join(set(other_tags))}。", "score": 0}

    think_pos = model_output.find("<think>")
    plan_pos = model_output.find("<plan>")
    python_pos = model_output.find("```python")
    if not (think_pos < plan_pos < python_pos):
        return {
            "reason": f"标签顺序错误。应为<think>→<plan>→```python，实际位置：<think>在{think_pos}，<plan>在{plan_pos}，```python在{python_pos}。",
            "score": 0,
        }

    return {"reason": "输出结构完整：包含且仅包含一次<think>...</think>、<plan>...</plan>和```python...```，顺序正确。", "score": 1}


def read_prompt(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None


def load_test_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None
    try:
        df = pd.read_csv(file_path).fillna("")
        print(f"成功从 {file_path} 加载了 {len(df)} 条测试数据。")
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return None


def construct_prompt(template: str, row: pd.Series, model_output: Optional[str] = None) -> str:
    prompt = template
    for col_name in ["question", "env", "hint", "memory", "dependency", "state"]:
        prompt = prompt.replace(f"{{{{ {col_name} }}}}", str(row.get(col_name, f"无[{col_name}]，请自行理解问题并给出答案")))
    if model_output is not None:
        prompt = prompt.replace("{{ label }}", model_output)
    return prompt


def construct_eval_prompt(template: str, row: pd.Series, model_output: str) -> str:
    prompt = template
    for col_name in ["question", "env", "hint", "memory", "dependency", "state", "GT1", "GT2"]:
        prompt = prompt.replace(f"{{{{ {col_name} }}}}", str(row.get(col_name, "")))
    prompt = prompt.replace("{{ label }}", model_output)
    return prompt


def parse_eval_json(eval_raw: str) -> Optional[dict]:
    """从评估模型输出中解析 JSON（兼容 ```json ...``` 或末尾 {...}）"""
    try:
        content = eval_raw
        json_match = re.search(r"```json\s*\n(.*?)\n```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            brace_match = re.search(r"\{[\s\S]*\}\s*$", content)
            if brace_match:
                content = brace_match.group(0)
        return json.loads(content)
    except Exception:
        return None


def recompute_is_match(eval_json: dict, other_threshold: int = 3) -> dict:
    """
    口径：result_equivalence=1 或 other_true_count>=other_threshold 判定匹配
    """
    try:
        ev = eval_json["details"]["evaluation"]
        result_correct = ev["result_equivalence"]["score"]
        other_keys = ["intent_accuracy", "plan_feasibility", "tool_selection", "param_validity", "step_efficiency", "structure_integrity"]

        other_true_count = 0
        for k in other_keys:
            if ev[k]["score"] == 1:
                other_true_count += 1

        if result_correct == 1 or other_true_count >= other_threshold:
            eval_json["is_match"] = True
            eval_json["reasons"] = "满足正确性要求，判定为匹配。"
        else:
            eval_json["is_match"] = False
            reasons = []
            for k in other_keys + ["result_equivalence"]:
                score = ev[k]["score"] if k != "result_equivalence" else ev["result_equivalence"]["score"]
                if score != 1:
                    reasons.append(f"{k}得分为{score}")
            eval_json["reasons"] = reasons

    except Exception:
        eval_json["is_match"] = False
        eval_json["reasons"] = ["eval_json结构异常"]

    return eval_json


# =========================
# Worker（单条样本）
# =========================

def process_one_item(
    index: int,
    row: pd.Series,
    *,
    main_client,
    eval_client,
    main_prompt_template: str,
    eval_prompt_template: str,
    n_samples: int,
    other_threshold: int = 3,
) -> Dict[str, Any]:
    """
    单条样本任务（线程池并发）：
    - greedy 1次 + eval 1次
    - sample n_samples 次 + eval n_samples 次
    返回：
    - results_rows: List[dict]
    - greedy_eval: dict
    - sample_evals: List[dict]
    - stats: 增量统计
    - timeout: bool（任一关键调用失败则 True）
    """

    timeout = False
    results_rows: List[dict] = []

    total_char_len = 0
    total_token_len = 0
    total_cot_char_len = 0
    total_cot_token_len = 0
    latencies_per_output: List[float] = []
    total_token_len_for_rs = 0

    greedy_eval: dict = {"is_match": False}
    sample_evals: List[dict] = []

    main_prompt = construct_prompt(main_prompt_template, row)

    # ===== greedy =====
    t0 = time.time()
    greedy_outputs = main_client.call_llm(main_prompt, temperature=0.0, top_p=1.0, n=1)
    t1 = time.time()
    greedy_latency = t1 - t0

    if not greedy_outputs:
        timeout = True
        return {
            "index": index,
            "timeout": True,
            "results_rows": [],
            "greedy_eval": {"is_match": False, "error": "greedy_call_failed"},
            "sample_evals": [],
            "stats": {
                "total_char_len": 0,
                "total_token_len": 0,
                "total_cot_char_len": 0,
                "total_cot_token_len": 0,
                "latencies_per_output": [],
                "total_token_len_for_rs": 0,
            },
        }

    greedy_output = greedy_outputs[0]

    char_len = len(greedy_output)
    token_len = main_client.count_tokens(greedy_output)
    total_char_len += char_len
    total_token_len += token_len
    total_token_len_for_rs += token_len
    latencies_per_output.append(greedy_latency)

    cot_match = re.search(r"<think>(.*?)</think>", greedy_output, re.DOTALL)
    cot_text = cot_match.group(1) if cot_match else greedy_output
    total_cot_char_len += len(cot_text)
    total_cot_token_len += main_client.count_tokens(cot_text)

    structure_result_greedy = validate_structure_integrity(greedy_output)

    eval_prompt_greedy = construct_eval_prompt(eval_prompt_template, row, greedy_output)
    eval_outs = eval_client.call_llm(eval_prompt_greedy, n=1, temperature=0.0, top_p=1.0)
    if eval_outs:
        parsed = parse_eval_json(eval_outs[0])
        if parsed is None:
            greedy_eval = {"is_match": False, "parse_error": "eval_json_parse_failed"}
        else:
            parsed.setdefault("details", {}).setdefault("evaluation", {})["structure_integrity"] = structure_result_greedy
            greedy_eval = recompute_is_match(parsed, other_threshold=other_threshold)
    else:
        greedy_eval = {"is_match": False, "error": "eval_call_failed"}

    results_rows.append({
        "index": index,
        "sample_id": 0,
        "sample_type": "greedy",
        "question": row.get("question", ""),
        "env": row.get("env", ""),
        "hint": row.get("hint", ""),
        "memory": row.get("memory", ""),
        "GT1": row.get("GT1", ""),
        "GT2": row.get("GT2", ""),
        "model_output": greedy_output,
        "evaluation": greedy_eval,
        "eval_raw_output": "",
    })

    # ===== samples (n_samples) =====
    t2 = time.time()
    sample_outputs = main_client.call_llm(
        main_prompt,
        temperature=main_client.temperature,
        top_p=main_client.top_p,
        n=n_samples,
    )
    t3 = time.time()
    sample_latency = t3 - t2

    if not sample_outputs:
        timeout = True
        # greedy 有结果也返回，方便统计
        return {
            "index": index,
            "timeout": True,
            "results_rows": results_rows,
            "greedy_eval": greedy_eval,
            "sample_evals": [],
            "stats": {
                "total_char_len": total_char_len,
                "total_token_len": total_token_len,
                "total_cot_char_len": total_cot_char_len,
                "total_cot_token_len": total_cot_token_len,
                "latencies_per_output": latencies_per_output,
                "total_token_len_for_rs": total_token_len_for_rs,
            },
        }

    per_sample_latency = sample_latency / len(sample_outputs) if len(sample_outputs) > 0 else sample_latency

    for i, sample_output in enumerate(sample_outputs, start=1):
        char_len = len(sample_output)
        token_len = main_client.count_tokens(sample_output)
        total_char_len += char_len
        total_token_len += token_len
        total_token_len_for_rs += token_len
        latencies_per_output.append(per_sample_latency)

        cot_match = re.search(r"<think>(.*?)</think>", sample_output, re.DOTALL)
        cot_text = cot_match.group(1) if cot_match else sample_output
        total_cot_char_len += len(cot_text)
        total_cot_token_len += main_client.count_tokens(cot_text)

        structure_result_sample = validate_structure_integrity(sample_output)

        eval_prompt_sample = construct_eval_prompt(eval_prompt_template, row, sample_output)
        eval_outs = eval_client.call_llm(eval_prompt_sample, n=1, temperature=0.0, top_p=1.0)

        if eval_outs:
            parsed = parse_eval_json(eval_outs[0])
            if parsed is None:
                eval_json = {"is_match": False, "parse_error": "eval_json_parse_failed"}
            else:
                parsed.setdefault("details", {}).setdefault("evaluation", {})["structure_integrity"] = structure_result_sample
                eval_json = recompute_is_match(parsed, other_threshold=other_threshold)
        else:
            eval_json = {"is_match": False, "error": "eval_call_failed"}

        sample_evals.append(eval_json)

        results_rows.append({
            "index": index,
            "sample_id": i,
            "sample_type": "sample",
            "question": row.get("question", ""),
            "env": row.get("env", ""),
            "hint": row.get("hint", ""),
            "memory": row.get("memory", ""),
            "GT1": row.get("GT1", ""),
            "GT2": row.get("GT2", ""),
            "model_output": sample_output,
            "evaluation": eval_json,
            "eval_raw_output": "",
        })

    return {
        "index": index,
        "timeout": timeout,
        "results_rows": results_rows,
        "greedy_eval": greedy_eval,
        "sample_evals": sample_evals,
        "stats": {
            "total_char_len": total_char_len,
            "total_token_len": total_token_len,
            "total_cot_char_len": total_cot_char_len,
            "total_cot_token_len": total_cot_token_len,
            "latencies_per_output": latencies_per_output,
            "total_token_len_for_rs": total_token_len_for_rs,
        },
    }


# =========================
# Main
# =========================

if __name__ == "__main__":
    MODE = "think"
    bi_prompt_pth = "./data/bi_prompt_new.txt"
    eval_prompt_pth = "./data/eval_prompt.txt"
    test_data_pth = "./data/hunhe_test.csv"
    output_results_pth = "./data/evaluation_14B_V24_prompt_hunhe_new_results.csv"
    output_summary_pth = "./data/evaluation_14B_V24_prompt_hunhe_new_summary.csv"

    WHICH_AS_MAIN = 0  # 0: 本地qwen; 1: ChatGPT; 2: DeepSeek
    N_SAMPLES = 3
    OTHER_THRESHOLD = 3  # 你原逻辑是 >=3

    model_name = "Qwen3-14B-V24_prompt_new"
    header = "bi测试集评测结果\n" f"模型: {model_name}\n"

    start_time = time.time()

    if WHICH_AS_MAIN == 1:
        main_client = OpenAIClient(model_name="gpt-4.1")
    elif WHICH_AS_MAIN == 0:
        main_client = LLMClient(ip="214.2.2.21", port="6094", model_name="qwen3_32b")
    else:
        main_client = DeepSeekClient(model_name="deepseek-reasoner")

    eval_client = LLMClient(ip="214.2.2.20", port="6191", model_name="qwen3_32b")

    main_prompt_template = read_prompt(bi_prompt_pth)
    eval_prompt_template = read_prompt(eval_prompt_pth)
    test_dataframe = load_test_data_from_csv(test_data_pth)

    if test_dataframe is None or main_prompt_template is None or eval_prompt_template is None:
        print("程序因无法加载测试数据或Prompt模板而终止。")
        raise SystemExit(1)

    # 并行度建议：先从 4~8 起步，避免压垮推理/评估服务
    num_items = len(test_dataframe)
    max_workers = min(8, num_items) if num_items > 0 else 1

    results: List[dict] = []
    timeout_item_indices = set()

    total_char_len = 0
    total_token_len = 0
    total_cot_char_len = 0
    total_cot_token_len = 0
    latencies_per_output: List[float] = []
    total_token_len_for_rs = 0

    greedy_eval_by_index: Dict[int, dict] = {}
    sample_eval_by_index: Dict[int, List[dict]] = {}

    # 主线程进度打印（不会交叉）
    done = 0
    progress_every = 10

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, row in test_dataframe.iterrows():
            futures.append(
                executor.submit(
                    process_one_item,
                    int(index),
                    row,
                    main_client=main_client,
                    eval_client=eval_client,
                    main_prompt_template=main_prompt_template,
                    eval_prompt_template=eval_prompt_template,
                    n_samples=N_SAMPLES,
                    other_threshold=OTHER_THRESHOLD,
                )
            )

        for fut in as_completed(futures):
            out = fut.result()
            idx = out["index"]

            if out.get("timeout"):
                timeout_item_indices.add(idx)

            results.extend(out["results_rows"])

            greedy_eval_by_index[idx] = out.get("greedy_eval", {"is_match": False})
            sample_eval_by_index[idx] = out.get("sample_evals", [])

            st = out["stats"]
            total_char_len += st["total_char_len"]
            total_token_len += st["total_token_len"]
            total_cot_char_len += st["total_cot_char_len"]
            total_cot_token_len += st["total_cot_token_len"]
            latencies_per_output.extend(st["latencies_per_output"])
            total_token_len_for_rs += st["total_token_len_for_rs"]

            done += 1
            if done % progress_every == 0 or done == num_items:
                print(f"[progress] {done}/{num_items} done")

    # 输出稳定性：按 index + sample_id 排序
    results.sort(key=lambda x: (x["index"], x["sample_id"]))

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
    print(f"总耗时: {elapsed_time_str} ({elapsed_seconds:.2f} 秒)")

    # ========= 统计 =========
    if not results:
        print("没有可用结果（可能全部超时/失败）。")
        raise SystemExit(0)

    total_count = len(results)
    correct_count = sum(1 for item in results if isinstance(item.get("evaluation"), dict) and item["evaluation"].get("is_match"))

    num_items = num_items
    expected_total_samples_for_passN = num_items * N_SAMPLES

    total_correct_samples_for_passN = 0
    num_items_with_at_least_one_pass = 0
    greedy_correct_items = 0

    for index, _row in test_dataframe.iterrows():
        idx = int(index)
        sample_evals = sample_eval_by_index.get(idx, [])
        correct_for_this_item = sum(1 for ev in sample_evals if isinstance(ev, dict) and ev.get("is_match"))
        total_correct_samples_for_passN += correct_for_this_item

        if correct_for_this_item > 0:
            num_items_with_at_least_one_pass += 1

        greedy_eval = greedy_eval_by_index.get(idx, {})
        if isinstance(greedy_eval, dict) and greedy_eval.get("is_match"):
            greedy_correct_items += 1

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    avg_char_len = total_char_len / total_count if total_count > 0 else 0.0
    avg_token_len = total_token_len / total_count if total_count > 0 else 0.0

    accuracy_passN = num_items_with_at_least_one_pass / num_items if num_items > 0 else 0.0
    total_passN = total_correct_samples_for_passN / expected_total_samples_for_passN if expected_total_samples_for_passN > 0 else 0.0
    stability_pass1 = greedy_correct_items / num_items if num_items > 0 else 0.0

    timeout_rate = len(timeout_item_indices) / num_items if num_items > 0 else 0.0

    cot_char_len_mean = total_cot_char_len / total_count if total_count > 0 else 0.0
    cot_token_len_mean = total_cot_token_len / total_count if total_count > 0 else 0.0

    def percentile(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        vals = sorted(vals)
        k = (len(vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        return vals[f] + (vals[c] - vals[f]) * (k - f)

    if latencies_per_output:
        latency_mean = sum(latencies_per_output) / len(latencies_per_output)
        latency_p50 = percentile(latencies_per_output, 0.5)
        latency_p95 = percentile(latencies_per_output, 0.95)
    else:
        latency_mean = latency_p50 = latency_p95 = 0.0

    total_latency_outputs = sum(latencies_per_output)
    RS = total_token_len_for_rs / total_latency_outputs if total_latency_outputs > 0 else 0.0

    print(f"总评估答案数(greedy+N): {total_count}, 正确数: {correct_count}, 准确率: {accuracy:.2%}")
    print(f"num_items(评测样本数): {num_items}")
    print(f"N(多样采样数): {N_SAMPLES}")
    print(f"accuracy_pass@N: {accuracy_passN:.2%}")
    print(f"total_pass@N: {total_passN:.2%} = {total_correct_samples_for_passN}/{expected_total_samples_for_passN}")
    print(f"stability_pass@1: {stability_pass1:.2%}")
    print(f"RS(token/s): {RS:.4f}")
    print(f"timeout_rate: {timeout_rate:.2%}")
    print(f"平均输出字符数: {avg_char_len:.2f}, 平均输出Token数: {avg_token_len:.2f}")
    print(f"cot_char_len_mean: {cot_char_len_mean:.2f}")
    print(f"cot_token_len_mean: {cot_token_len_mean:.2f}")
    print(f"latency_mean: {latency_mean:.4f}")
    print(f"latency_p50: {latency_p50:.4f}")
    print(f"latency_p95: {latency_p95:.4f}")

    # 写 CSV 前，把 evaluation 转成字符串
    results_to_save = []
    for item in results:
        item_copy = dict(item)
        item_copy["evaluation"] = json.dumps(item_copy.get("evaluation", {}), ensure_ascii=False)
        results_to_save.append(item_copy)

    df_results = pd.DataFrame(results_to_save)
    df_results.to_csv(output_results_pth, encoding="utf-8-sig", index=False)
    print(f"评估结果已保存到 {output_results_pth}")

    summary = {
        "mode": MODE,
        "N": N_SAMPLES,
        "num_items": num_items,

        "total_count": total_count,
        "correct_count": correct_count,
        "accuracy": accuracy,

        "accuracy_pass@N": accuracy_passN,
        "total_pass@N": total_passN,
        "stability_pass@1": stability_pass1,

        "RS": RS,
        "timeout_rate": timeout_rate,

        "avg_char_len": avg_char_len,
        "avg_token_len": avg_token_len,
        "cot_char_len_mean": cot_char_len_mean,
        "cot_token_len_mean": cot_token_len_mean,

        "latency_mean": latency_mean,
        "latency_p50": latency_p50,
        "latency_p95": latency_p95,

        "elapsed_seconds": elapsed_seconds,
        "elapsed_time_str": elapsed_time_str,
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(output_summary_pth, encoding="utf-8-sig", index=False)
    print(f"评估汇总已保存到 {output_summary_pth}")

    # 飞书通知
    lark_callback.send_msg(
        content=(
            header
            + f"num_items（评测样本数）: {num_items}\n"
            + f"N(多样采样数): {N_SAMPLES}\n"
            + f"total_count（答案数）: {total_count}\n"
            + f"correct_count（正确答案数）: {correct_count}\n"
            + f"accuracy（准确率）: {accuracy:.2%}\n"
            + f"accuracy_pass@N（至少一个正确的样本占比）: {accuracy_passN:.2%}\n"
            + f"total_pass@N（总正确数 / (N*num_items)）: {total_passN:.2%} = {total_correct_samples_for_passN}/{expected_total_samples_for_passN}\n"
            + f"stability_pass@1（greedy正确率）: {stability_pass1:.2%}\n"
            + f"RS(token/s): {RS:.4f}\n"
            + f"timeout_rate: {timeout_rate:.2%}\n"
            + f"cot_char_len_mean: {cot_char_len_mean:.2f}\n"
            + f"cot_token_len_mean: {cot_token_len_mean:.2f}\n"
            + f"latency_mean: {latency_mean:.4f}\n"
            + f"latency_p50: {latency_p50:.4f}\n"
            + f"latency_p95: {latency_p95:.4f}\n"
            + f"elapsed_time: {elapsed_time_str}"
        )
    )
