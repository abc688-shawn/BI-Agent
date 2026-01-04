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

'''
脚本备注
- 包含对通用数据和北银数据的正确率统计
'''

# 飞书机器人配置 - 自动化评估bot
lark_callback = LarkCallback(
    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/986352c7-a346-4e33-9126-c7fd9208b5a0",
    secret="lwSomSfbR4XZpX33cIhWYb",
)

qwen_tokenizer = AutoTokenizer.from_pretrained(
    "/data01/lhs/mycode/output/agent/Qwen3_14B/dense_agent_v1_bi_think/hf_output"
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


if __name__ == "__main__":
    # 评测模式（think / no_think）
    MODE = "think"
    bi_prompt_pth = "data/prompt/bi_prompt.txt"
    eval_prompt_pth = "data/prompt/eval_prompt_new.txt"
    test_data_pth = "data/test_all_bi_beiyin.csv"
    output_results_pth = "output/evaluation_14B_V24_results.csv"
    output_summary_pth = "output/evaluation_14B_V24_summary.csv"
    WHICH_AS_MAIN = 0  # 0: 用本地 qwen; 1: ChatGPT; 2: DeepSeek

    model_name = "Qwen3-14B-V24-only for bi"
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
            port="6094",
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
        beiyin_succ = 0
        beiyin_count = 0
        norm_succ = 0
        norm_count = 0 

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

        for index, row in test_dataframe.iterrows():
            print(f"\n{'=' * 20} 正在处理第 {index + 1} 条数据 {'=' * 20}")

            main_prompt = construct_prompt(main_prompt_template, row)

            # ========= greedy 模式 =========
            print("[发送 greedy 模式请求]...")
            greedy_start = time.time()
            greedy_outputs = main_client.call_llm(
                main_prompt,
                temperature=0.0,  # 近似 greedy
                top_p=1.0,
                n=1,
            )
            greedy_end = time.time()
            greedy_latency = greedy_end - greedy_start

            if greedy_outputs:
                greedy_output = greedy_outputs[0]
                print(f"[greedy 模型输出]:\n{greedy_output}\n")

                # 记录长度 & token
                char_len = len(greedy_output)
                token_len = main_client.count_tokens(greedy_output)
                total_char_len += char_len
                total_token_len += token_len
                total_token_len_for_rs += token_len

                # CoT：取 <think>...</think>，没有就取全输出
                cot_match = re.search(r"<think>(.*?)</think>", greedy_output, re.DOTALL)
                if cot_match:
                    cot_text = cot_match.group(1)
                else:
                    cot_text = greedy_output
                cot_char_len = len(cot_text)
                cot_token_len = main_client.count_tokens(cot_text)
                total_cot_char_len += cot_char_len
                total_cot_token_len += cot_token_len

                # latency 记录
                latencies_per_output.append(greedy_latency)

                # 结构验证
                structure_result_greedy = validate_structure_integrity(greedy_output)
                print(
                    f"[greedy 结构验证]: {structure_result_greedy['reason']} "
                    f"(得分: {structure_result_greedy['score']})\n"
                )

                # 调用评估模型
                eval_prompt_greedy = construct_eval_prompt(eval_prompt_template, row, greedy_output)
                print("[发送 greedy 评估打分请求]...")
                evaluation_outputs_greedy = eval_client.call_llm(
                    eval_prompt_greedy,
                    n=1,
                    temperature=0.0,  # 评估建议用确定性输出
                    top_p=1.0,
                )

                if evaluation_outputs_greedy:
                    eval_raw = evaluation_outputs_greedy[0]
                    try:
                        eval_content = eval_raw
                        json_match = re.search(r"```json\s*\n(.*?)\n```", eval_content, re.DOTALL)
                        if json_match:
                            eval_content = json_match.group(1)
                        else:
                            # 没有代码块时，从最后一个 '{' 开始截到最后一个 '}'，粗暴抠 JSON
                            brace_match = re.search(r"\{[\s\S]*\}\s*$", eval_content)
                            if brace_match:
                                eval_content = brace_match.group(0)

                        eval_json = json.loads(eval_content)

                        # 合并结构验证结果
                        eval_json["details"]["evaluation"]["structure_integrity"] = structure_result_greedy

                        # 重新计算 is_match
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
                            eval_json["is_match"] = True
                            eval_json["reasons"] = "满足正确性要求，判定为匹配。"
                        else:
                            eval_json["is_match"] = False
                            reasons = []
                            for key in other_keys + ["result_equivalence"]:
                                score = eval_json["details"]["evaluation"][key]["score"]
                                if score != 1:
                                    reasons.append(f"{key}得分为{score}")
                            eval_json["reasons"] = reasons

                        print(
                            f"[greedy 最终评估结果]:\n"
                            f"{json.dumps(eval_json, ensure_ascii=False, indent=2)}"
                        )

                        # 保存到 results（sample_type=greedy, sample_id=0）
                        results.append(
                            {
                                "index": index,
                                "sample_id": 0,
                                "sample_type": "greedy",
                                "type": row.get("type", ""),
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
                        )

                    except json.JSONDecodeError as e:
                        print(f"[greedy 评估结果解析失败]: {str(e)}")
                        print(f"原始输出: {evaluation_outputs_greedy[0]}")
                        # 即便解析失败，也记录一条结果，evaluation 标记错误信息
                        results.append(
                            {
                                "index": index,
                                "sample_id": 0,
                                "sample_type": "greedy",
                                "type": row.get("type", ""),
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
                        )
                    except Exception as e:
                        print(f"[处理 greedy 评估结果时出错]: {str(e)}")
                        results.append(
                            {
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
                                "evaluation": {"error": f"Exception: {str(e)}"},
                                "eval_raw_output": eval_raw,
                            }
                        )
                else:
                    print("[greedy 评估失败]: 未能获取评估结果。")
            else:
                print("[greedy 模型调用失败]: 未能获取模型输出。")
                timeout_item_indices.add(index)

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
                if item["evaluation"].get("is_match"):
                    correct_count += 1
                if item["type"] == "北银":
                    beiyin_count += 1
                    if item["evaluation"].get("is_match"):
                        beiyin_succ += 1
                if item["type"] == "通用":
                    norm_count += 1
                    if item["evaluation"].get("is_match"):
                        norm_succ += 1

            beiyin_succ_rate = beiyin_succ / beiyin_count if beiyin_count > 0 else 0.0
            norm_succ_rate = norm_succ / norm_count if norm_count > 0 else 0.0

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
            print(f"北银样本数: {beiyin_count}, 北银正确数: {beiyin_succ}, 北银准确率: {beiyin_succ_rate:.2%}")
            print(f"通用样本数: {norm_count}, 通用正确数: {norm_succ}, 通用准确率: {norm_succ_rate:.2%}")
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

            summary = {
                "mode": MODE,  # think / no_think
                "num_items": num_items,  # 评估样本数

                "total_count": total_count,  # greedy 答案数
                "correct_count": correct_count,  # greedy 正确数量
                "accuracy": accuracy,  # greedy 整体准确率

                "RS": RS,  # token per second
                "timeout_rate": timeout_rate,  # 超时率（有调用失败的样本占比）

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

            # 飞书通知（只保留 greedy 指标）
            lark_callback.send_msg(
                content=(
                    header
                    + f"num_items（评测样本数）: {num_items}\n"
                    + f"total_count（greedy 答案数）: {total_count}\n"
                    + f"correct_count（正确答案数）: {correct_count}\n"
                    + f"accuracy（greedy 准确率）: {accuracy:.2%}\n"
                    + f"北银样本数: {beiyin_count}, 北银正确数: {beiyin_succ}, 北银准确率: {beiyin_succ_rate:.2%}\n"
                    + f"通用样本数: {norm_count}, 通用正确数: {norm_succ}, 通用准确率: {norm_succ_rate:.2%}\n"
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
        print("程序因无法加载测试数据或Prompt模板而终止。")
