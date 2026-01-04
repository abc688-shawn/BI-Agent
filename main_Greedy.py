import requests
import time
import math
import os
import json
import re
from typing import Optional, List, Dict
import pandas as pd
from swanlab.plugin.notification import LarkCallback
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

'''
脚本备注
- 去除对通用数据和北银数据的正确率统计
- 使用线程池并发加速评估（每条样本：主模型->结构验证->评估模型->解析->记录）
'''

# 飞书机器人配置 - 自动化评估bot
lark_callback = LarkCallback(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/986352c7-a346-4e33-9126-c7fd9208b5a0",
    secret="lwSomSfbR4XZpX33cIhWYb",
)

qwen_tokenizer = AutoTokenizer.from_pretrained(
    "/data01/lhs/mycode/output/agent/Qwen3_14B/dense_agent_v2_bi_think/hf_output"
)

try:
    gpt4_encoding = tiktoken.encoding_for_model("gpt-4.1")
except Exception:
    # 不认识模型名时退回通用 cl100k_base
    gpt4_encoding = tiktoken.get_encoding("cl100k_base")

try:
    deepseek_reasoner_encoding = tiktoken.get_encoding("deepseek-reasoner")
except Exception:
    deepseek_reasoner_encoding = tiktoken.get_encoding("cl100k_base")

try:
    deepseek_chat_encoding = tiktoken.get_encoding("deepseek-chat")
except Exception:
    deepseek_chat_encoding = tiktoken.get_encoding("cl100k_base")


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
                else:
                    print(f"第{attempt + 1}次调用失败，正在重试: {str(e)}")
                    time.sleep(2)
                    continue
            except Exception as e:
                print(f"未知错误: {str(e)}")
                return None

    def count_tokens(self, text: str) -> int:
        """
        使用本地 Qwen tokenizer 统计 token 数量
        """
        if not text:
            return 0
        input_ids = qwen_tokenizer.encode(text)
        return len(input_ids)


class OpenAIClient:
    """
    使用 OpenAI ChatGPT 的客户端，接口仿照 LLMClient.call_llm：
    - call_llm(prompt, temperature, top_p, n) -> List[str]
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.temperature = 0.7
        self.top_p = 0.9
        self.n = 1

        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            # 不支持的模型名回退到通用 cl100k_base
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

                # 返回 List[str]，与原来的 LLMClient 保持一致
                return [choice.message.content for choice in chat_completion.choices]

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"调用 OpenAI API 失败（已重试 {max_retries} 次）: {str(e)}")
                    return None
                else:
                    print(f"第 {attempt + 1} 次调用 OpenAI 失败，正在重试: {str(e)}")
                    time.sleep(2)
                    continue

    def count_tokens(self, text: str) -> int:
        """
        使用 tiktoken 统计 OpenAI 模型的 token 数量
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))


class DeepSeekClient:
    """
    使用 DeepSeek 作为主模型，接口仿照 LLMClient / OpenAIClient：
    - call_llm(prompt, temperature, top_p, n) -> List[str]
    """

    def __init__(self, model_name: str = "deepseek-chat"):
        self.model_name = model_name
        self.temperature = 0.7
        self.top_p = 0.9
        self.n = 1

        # 根据模型名选择对应 encoding
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
                        # reasoning_content 在 message.model_extra 里
                        reasoning = ""
                        if hasattr(msg, "model_extra") and isinstance(msg.model_extra, dict):
                            reasoning = msg.model_extra.get("reasoning_content", "") or ""

                        final_content = msg.content or ""

                        content = f"<think>\n{reasoning}\n</think>\n{final_content}"
                    else:
                        # 非 reasoner 模型原样返回，避免影响 eval 的 deepseek-chat
                        content = msg.content or ""

                    outputs.append(content)
                return outputs

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"调用 DeepSeek API 失败（已重试 {max_retries} 次）: {str(e)}")
                    return None
                else:
                    print(f"第 {attempt + 1} 次调用 DeepSeek 失败，正在重试: {str(e)}")
                    time.sleep(2)
                    continue

    def count_tokens(self, text: str) -> int:
        """
        使用 DeepSeek 对应 tokenizer 统计 token 数量
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))


def validate_structure_integrity(model_output: str) -> Dict[str, any]:
    """使用正则表达式验证模型输出的结构完整性"""
    task_completed_pattern = r"当前任务已完成，无需工具调用|任务已完成|无需工具调用"
    is_task_completed = re.search(task_completed_pattern, model_output) is not None

    if is_task_completed:
        think_count = len(re.findall(r"<think>", model_output))
        think_close_count = len(re.findall(r"</think>", model_output))
        plan_count = len(re.findall(r"<plan>", model_output))
        python_count = len(re.findall(r"```python", model_output))

        if think_count == 1 and think_close_count == 1 and plan_count == 0 and python_count == 0:
            return {
                "reason": "任务已完成的特例：正确包含一次<think>...</think>和完成说明，无其他结构。",
                "score": 1,
            }
        else:
            return {
                "reason": f"任务已完成的特例格式错误：<think>出现{think_count}次，</think>出现{think_close_count}次，<plan>出现{plan_count}次，```python出现{python_count}次。",
                "score": 0,
            }

    think_pattern = r"<think>.*?</think>"
    plan_pattern = r"<plan>.*?</plan>"
    python_pattern = r"```python.*?```"

    think_matches = re.findall(think_pattern, model_output, re.DOTALL)
    plan_matches = re.findall(plan_pattern, model_output, re.DOTALL)
    python_matches = re.findall(python_pattern, model_output, re.DOTALL)

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
        return {
            "reason": f"```python...```代码块应出现1次，实际出现{len(python_matches)}次。",
            "score": 0,
        }

    if other_tags:
        return {
            "reason": f"输出包含不允许的标签：{', '.join(set(other_tags))}。",
            "score": 0,
        }

    think_pos = model_output.find("<think>")
    plan_pos = model_output.find("<plan>")
    python_pos = model_output.find("```python")

    if not (think_pos < plan_pos < python_pos):
        return {
            "reason": f"标签顺序错误。应为<think>→<plan>→```python，实际位置：<think>在{think_pos}，<plan>在{plan_pos}，```python在{python_pos}。",
            "score": 0,
        }

    return {
        "reason": "输出结构完整：包含且仅包含一次<think>...</think>、<plan>...</plan>和```python...```，顺序正确。",
        "score": 1,
    }


def read_prompt(file_path: str) -> str:
    """从文件读取prompt"""
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
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
        df = pd.read_csv(file_path).fillna("")
        print(f"成功从 {file_path} 加载了 {len(df)} 条测试数据。")
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return None


def construct_prompt(template: str, row: pd.Series, model_output: str = None) -> str:
    """根据DataFrame的一行数据和模板，构建prompt。"""
    prompt = template
    for col_name in ["question", "env", "hint", "memory", "dependency", "state"]:
        prompt = prompt.replace(f"{{{{ {col_name} }}}}", str(row.get(col_name, f"无[{col_name}]，请自行理解问题并给出答案")))
    if model_output is not None:
        prompt = prompt.replace("{{ label }}", model_output)
    return prompt


def construct_eval_prompt(template: str, row: pd.Series, model_output: str) -> str:
    """为评估模型构建prompt"""
    prompt = template
    for col_name in ["question", "env", "hint", "memory", "dependency", "state", "GT1", "GT2"]:
        prompt = prompt.replace(f"{{{{ {col_name} }}}}", str(row.get(col_name, "")))
    prompt = prompt.replace("{{ label }}", model_output)
    return prompt


def process_one_item(index: int,row: pd.Series,*,main_client,eval_client,main_prompt_template,eval_prompt_template,print_lock,) -> Dict[str, any]:
    """
    单条样本：主模型 greedy -> 结构验证 -> 评估模型 -> 解析 -> 返回结果与统计
    - 默认不打印模型全文输出，避免刷屏与串行开销
    """
    # ===== 可调的日志开关 =====
    VERBOSE = False              # True: 打印更多细节（仍然是一次性打印）
    PRINT_MODEL_OUTPUT = False   # True: 打印 greedy_output

    log_lines = []
    def log(msg: str):
        log_lines.append(msg)

    try:
        log(f"\n{'=' * 20} 正在处理第 {index + 1} 条数据 {'=' * 20}")

        main_prompt = construct_prompt(main_prompt_template, row)

        # ========= greedy 模式 =========
        t0 = time.time()
        greedy_outputs = main_client.call_llm(
            main_prompt,
            temperature=0.0,
            top_p=1.0,
            n=1,
        )
        t1 = time.time()
        greedy_latency = t1 - t0

        if not greedy_outputs:
            log(f"[greedy FAIL] index={index} latency={greedy_latency:.3f}s (no output)")
            with print_lock:
                print("\n".join(log_lines))
            return {"ok": False, "index": index, "timeout": True}

        greedy_output = greedy_outputs[0]

        # 可选：打印模型输出（默认关闭）
        if PRINT_MODEL_OUTPUT:
            # 只打前后片段，避免超长输出
            head = greedy_output[:800]
            tail = greedy_output[-800:] if len(greedy_output) > 800 else ""
            if tail:
                log(f"[greedy output head]\n{head}\n...[snip]...\n[greedy output tail]\n{tail}\n")
            else:
                log(f"[greedy output]\n{head}\n")

        # 记录长度 & token
        char_len = len(greedy_output)
        token_len = main_client.count_tokens(greedy_output)

        # CoT：取 <think>...</think>，没有就取全输出
        cot_match = re.search(r"<think>(.*?)</think>", greedy_output, re.DOTALL)
        cot_text = cot_match.group(1) if cot_match else greedy_output
        cot_char_len = len(cot_text)
        cot_token_len = main_client.count_tokens(cot_text)

        # 结构验证
        structure_result_greedy = validate_structure_integrity(greedy_output)

        if VERBOSE:
            log(f"[greedy OK] latency={greedy_latency:.3f}s char={char_len} tok={token_len}")
            log(f"[structure] score={structure_result_greedy['score']} reason={structure_result_greedy['reason']}")

        # ========= eval 模型 =========
        eval_prompt_greedy = construct_eval_prompt(eval_prompt_template, row, greedy_output)

        t2 = time.time()
        evaluation_outputs_greedy = eval_client.call_llm(
            eval_prompt_greedy,
            n=1,
            temperature=0.0,
            top_p=1.0,
        )
        t3 = time.time()
        eval_latency = t3 - t2

        if not evaluation_outputs_greedy:
            log(f"[eval FAIL] index={index} eval_latency={eval_latency:.3f}s (no output)")
            result_item = {
                "index": index,
                "sample_id": 0,
                "sample_type": "greedy",
                "type": row.get("type", "未规定"),
                "question": row.get("question", ""),
                "env": row.get("env", ""),
                "hint": row.get("hint", ""),
                "memory": row.get("memory", ""),
                "GT1": row.get("GT1", ""),
                "GT2": row.get("GT2", ""),
                "model_output": greedy_output,
                "evaluation": {"error": "eval_call_failed"},
                "eval_raw_output": "",
            }
            with print_lock:
                print("\n".join(log_lines))
            return {
                "ok": True,
                "timeout": False,
                "result_item": result_item,
                "char_len": char_len,
                "token_len": token_len,
                "cot_char_len": cot_char_len,
                "cot_token_len": cot_token_len,
                "latency": greedy_latency,
                "is_match": False,
            }

        eval_raw = evaluation_outputs_greedy[0]

        # 解析评估 JSON
        try:
            eval_content = eval_raw
            json_match = re.search(r"```json\s*\n(.*?)\n```", eval_content, re.DOTALL)
            if json_match:
                eval_content = json_match.group(1)
            else:
                brace_match = re.search(r"\{[\s\S]*\}\s*$", eval_content)
                if brace_match:
                    eval_content = brace_match.group(0)

            eval_json = json.loads(eval_content)

            # 合并结构验证结果
            eval_json.setdefault("details", {}).setdefault("evaluation", {})["structure_integrity"] = structure_result_greedy

            # 计算 is_match
            result_correct = eval_json["details"]["evaluation"]["result_equivalence"]["score"]
            other_keys = [
                "intent_accuracy",
                "plan_feasibility",
                "tool_selection",
                "param_validity",
                "step_efficiency",
                "structure_integrity",
            ]
            other_true_count = 0
            for key in other_keys:
                score = eval_json["details"]["evaluation"][key]["score"]
                if score == 1:
                    other_true_count += 1

            if result_correct == 1 or other_true_count >= 5:
                is_match_val = True
                reasons_val = "满足正确性要求，判定为匹配。"
            else:
                is_match_val = False
                reasons_list = []
                for key in other_keys + ["result_equivalence"]:
                    score = eval_json["details"]["evaluation"][key]["score"]
                    if score != 1:
                        reasons_list.append(f"{key}得分为{score}")
                reasons_val = reasons_list

            # 重构 eval_json 顺序，将 is_match 和 reasons 放在最前面
            new_eval_json = {"is_match": is_match_val, "reasons": reasons_val}
            new_eval_json.update(eval_json)
            eval_json = new_eval_json

            is_match = is_match_val

            # 最终仅输出一行汇总（默认）
            log(f"[DONE] index={index} greedy={greedy_latency:.3f}s eval={eval_latency:.3f}s match={is_match} "
                f"tok={token_len} cot_tok={cot_token_len} struct={structure_result_greedy['score']}")

            if VERBOSE:
                log(f"[eval raw head] {eval_raw[:500]}{'...' if len(eval_raw) > 500 else ''}")

            result_item = {
                "index": index,
                "sample_id": 0,
                "sample_type": "greedy",
                "type": row.get("type", "未规定"),
                "question": row.get("question", ""),
                "env": row.get("env", ""),
                "hint": row.get("hint", ""),
                "memory": row.get("memory", ""),
                "GT1": row.get("GT1", ""),
                "GT2": row.get("GT2", ""),
                "model_output": greedy_output,
                "evaluation": eval_json,
                "eval_raw_output": "",
            }

            # 一次性打印，避免交叉
            with print_lock:
                print("\n".join(log_lines))

            return {
                "ok": True,
                "timeout": False,
                "result_item": result_item,
                "char_len": char_len,
                "token_len": token_len,
                "cot_char_len": cot_char_len,
                "cot_token_len": cot_token_len,
                "latency": greedy_latency,
                "is_match": is_match,
            }

        except json.JSONDecodeError as e:
            log(f"[eval PARSE FAIL] index={index} err={str(e)}")
            if VERBOSE:
                log(f"[eval raw head]\n{eval_raw[:1200]}\n")

            result_item = {
                "index": index,
                "sample_id": 0,
                "sample_type": "greedy",
                "type": row.get("type", "未规定"),
                "question": row.get("question", ""),
                "env": row.get("env", ""),
                "hint": row.get("hint", ""),
                "memory": row.get("memory", ""),
                "GT1": row.get("GT1", ""),
                "GT2": row.get("GT2", ""),
                "model_output": greedy_output,
                "evaluation": {"parse_error": f"JSONDecodeError: {str(e)}"},
                "eval_raw_output": eval_raw,
            }

            with print_lock:
                print("\n".join(log_lines))

            return {
                "ok": True,
                "timeout": False,
                "result_item": result_item,
                "char_len": char_len,
                "token_len": token_len,
                "cot_char_len": cot_char_len,
                "cot_token_len": cot_token_len,
                "latency": greedy_latency,
                "is_match": False,
            }

    except Exception as e:
        log(f"[process_one_item EXCEPTION] index={index} err={str(e)}")
        with print_lock:
            print("\n".join(log_lines))
        return {"ok": False, "index": index, "timeout": False, "error": str(e)}


if __name__ == "__main__":
    # 评测模式（think / no_think）
    MODE = "think"
    bi_prompt_pth = "data/prompt/bi_prompt_v5.md"
    eval_prompt_pth = "data/prompt/eval_prompt_v3.md"
    test_data_pth = "data/tongyong_test.csv"
    output_results_pth = "output/evaluation_14BV29_biv5_for_tongyong_results.csv"
    WHICH_AS_MAIN = 0  # 0: 用本地 qwen; 1: ChatGPT; 2: DeepSeek
    model_name = "qwen3_14b_V29_biv5"
    feishu_yes = False # 是否启用飞书通知
    header = (
        "bi测试集评测结果\n"
        f"模型: {model_name}\n"
        f"模式: greedy\n"
    )

    # 开始记录评测的开始时间
    start_time = time.time()

    if WHICH_AS_MAIN == 1:
        main_client = OpenAIClient(
            model_name="gpt-4.1"  # "gpt-4.1-mini" 或 "gpt-4.1"
        )
    elif WHICH_AS_MAIN == 0:
        main_client = LLMClient(
            ip="214.2.2.21",
            port="6091",
            model_name="qwen3_32b",
        )
    elif WHICH_AS_MAIN == 2:
        main_client = DeepSeekClient(
            model_name="deepseek-reasoner"
        )

    eval_client = LLMClient(
        ip="214.2.2.20",
        port="6191",
        model_name="qwen3_32b",
    )

    main_prompt_template = read_prompt(bi_prompt_pth)
    eval_prompt_template = read_prompt(eval_prompt_pth)
    test_dataframe = load_test_data_from_csv(test_data_pth)

    if test_dataframe is not None and main_prompt_template is not None and eval_prompt_template is not None:
        results = []

        # 统计（只对 greedy 输出）
        total_char_len = 0
        total_token_len = 0

        # CoT 统计（只对 greedy 输出）
        total_cot_char_len = 0
        total_cot_token_len = 0

        latencies_per_output = []      # 每个 greedy 答案的耗时
        total_token_len_for_rs = 0     # 用于 token/s

        # timeout 统计（按“样本行”统计）
        timeout_item_indices = set()

        total_items_expected = len(test_dataframe)  # num_items

        # 为了让多线程日志不串
        print_lock = Lock()

        # ===================== 线程池并发执行 =====================
        num_items = total_items_expected

        # 线程数建议：I/O 密集型可较大，但避免压垮后端服务
        max_workers = min(8, num_items)  # 可按你的服务能力调大/调小（如 8/16/32）

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for index, row in test_dataframe.iterrows():
                futures.append(executor.submit(
                    process_one_item, 
                    int(index), 
                    row, 
                    main_client=main_client,
                    eval_client=eval_client,
                    main_prompt_template=main_prompt_template,
                    eval_prompt_template=eval_prompt_template,
                    print_lock=print_lock))

            for fut in as_completed(futures):
                out = fut.result()

                if not out.get("ok", False):
                    if out.get("timeout"):
                        timeout_item_indices.add(out.get("index"))
                    continue

                if out.get("timeout"):
                    timeout_item_indices.add(out["result_item"]["index"])
                    continue

                item = out["result_item"]
                results.append(item)

                total_char_len += out["char_len"]
                total_token_len += out["token_len"]
                total_cot_char_len += out["cot_char_len"]
                total_cot_token_len += out["cot_token_len"]
                latencies_per_output.append(out["latency"])
                total_token_len_for_rs += out["token_len"]

        # 为了输出与保存更稳定，按 index 排序
        results.sort(key=lambda x: (x["index"], x["sample_id"]))

        print(f"\n{'=' * 20} 所有数据处理完毕 {'=' * 20}")

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
        print(f"总耗时: {elapsed_time_str} ({elapsed_seconds:.2f} 秒)")

        # ========= 统计各类指标（仅 greedy） =========
        if results:
            # total_count：greedy 答案数量（通常等于样本数减去超时）
            total_count = len(results)
            correct_count = 0

            num_items = total_items_expected  # 评测样本数

            # accuracy：greedy 整体准确率
            for item in results:
                # item["evaluation"] 在写CSV前会转字符串；此时仍是dict
                if isinstance(item.get("evaluation"), dict) and item["evaluation"].get("is_match"):
                    correct_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0.0
            avg_char_len = total_char_len / total_count if total_count > 0 else 0.0
            avg_token_len = total_token_len / total_count if total_count > 0 else 0.0

            # timeout_rate：有调用失败的样本占比
            timeout_rate = len(timeout_item_indices) / num_items if num_items > 0 else 0.0

            # CoT 均值（对所有 greedy 答案）
            cot_char_len_mean = total_cot_char_len / total_count if total_count > 0 else 0.0
            cot_token_len_mean = total_cot_token_len / total_count if total_count > 0 else 0.0

            # latency（对每个 greedy 答案）
            if latencies_per_output:
                def percentile(vals, p):
                    vals = sorted(vals)
                    if not vals:
                        return 0.0
                    k = (len(vals) - 1) * p
                    f = math.floor(k)
                    c = math.ceil(k)
                    if f == c:
                        return vals[int(k)]
                    return vals[f] + (vals[c] - vals[f]) * (k - f)

                latency_mean = sum(latencies_per_output) / len(latencies_per_output)
                latency_p50 = percentile(latencies_per_output, 0.5)
                latency_p95 = percentile(latencies_per_output, 0.95)
            else:
                latency_mean = latency_p50 = latency_p95 = 0.0

            # RS：token per second（只用模型生成时间）
            total_latency_outputs = sum(latencies_per_output)
            RS = total_token_len_for_rs / total_latency_outputs if total_latency_outputs > 0 else 0.0

            print(f"总评估答案数(greedy): {total_count}, 正确数: {correct_count}, 准确率: {accuracy:.2%}")
            print(f"num_items(评测样本数): {num_items}")
            print(f"RS(token/s): {RS:.4f}")
            print(f"timeout_rate: {timeout_rate:.2%}")
            print(f"平均输出字符数: {avg_char_len:.2f}, 平均输出Token数: {avg_token_len:.2f}")
            print(f"cot_char_len_mean: {cot_char_len_mean:.2f}")
            print(f"cot_token_len_mean: {cot_token_len_mean:.2f}")
            print(f"latency_mean: {latency_mean:.4f}")
            print(f"latency_p50: {latency_p50:.4f}")
            print(f"latency_p95: {latency_p95:.4f}")

            # 写 CSV 前，把 evaluation 转成字符串
            for item in results:
                item["evaluation"] = json.dumps(item["evaluation"], ensure_ascii=False)

            df_results = pd.DataFrame(results)
            df_results.to_csv(output_results_pth, encoding="utf-8-sig", index=False)
            print(f"评估结果已保存到 {output_results_pth}")

            if feishu_yes:
                lark_callback.send_msg(
                    content=(
                        header
                        + f"num_items（评测样本数）: {num_items}\n"
                        + f"total_count（greedy 答案数）: {total_count}\n"
                        + f"correct_count（正确答案数）: {correct_count}\n"
                        + f"accuracy（greedy 准确率）: {accuracy:.2%}\n"
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

        else:
            print("没有可用结果（可能全部超时/失败）。")

    else:
        print("程序因无法加载测试数据或Prompt模板而终止。")
