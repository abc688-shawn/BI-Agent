import requests
import time
import os
from typing import Optional, List
import pandas as pd

class LLMClient:
    def __init__(self, ip: str, port: str, model_name: str):
        self.ip = ip
        self.port = port
        self.model_name = model_name
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.1
        self.n = 1
    
    
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
            "max_tokens": 4096
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120) # 增加超时时间
                response.raise_for_status()
                data = response.json()
                choices = data.get('choices', [])
                return [choice['message']['content'] for choice in choices]
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"调用API失败（已重试{max_retries}次）: {str(e)}")
                    return None
                else:
                    print(f"第{attempt + 1}次调用失败，正在重试: {str(e)}")
                    time.sleep(2) # 增加重试等待时间
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
            return file.read() # 读取完整文件，不strip()
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

def load_test_data_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """从 CSV 文件加载测试数据。"""
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None
    try:
        df = pd.read_csv(file_path).fillna('') # 使用fillna('')避免NaN导致的问题
        print(f"成功从 {file_path} 加载了 {len(df)} 条测试数据。")
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return None


def construct_eval_prompt(template: str, row: pd.Series, model_output: str) -> str:
    """为评估模型构建prompt"""
    prompt = template
    for col_name in ['question', 'env', 'hint', 'memory', 'dependency', 'state']:
        prompt = prompt.replace(f'{{{{ {col_name} }}}}', str(row.get(col_name, '')))
    prompt = prompt.replace('{{ label }}', model_output)
    return prompt


if __name__ == "__main__":
    # --- 步骤 1: 初始化客户端和加载数据 ---
    eval_client = LLMClient(
        ip="214.2.2.21",
        port="6095",
        model_name="qwen3_32b"
    )
    
    # 仅加载评估所需的模板
    eval_prompt_template = read_prompt('eval_prompt.txt')
    
    test_dataframe = load_test_data_from_csv('/home/shawn/mycode/train_data.csv')
    
    if test_dataframe is not None and eval_prompt_template is not None:
        for index, row in test_dataframe.iterrows():
            print(f"\n{'='*20} 正在处理第 {index + 1} 条数据 {'='*20}")
            
            # --- 步骤 2: 获取待评估的Label ---
            # 直接从CSV中读取label，不再调用主模型生成
            model_output = str(row.get('label', ''))
            
            if not model_output or model_output.lower() == 'nan':
                print("[警告]: 该行数据缺少 'label' 字段或为空，跳过评估。")
                continue

            print(f"[待评估的Label]:\n{model_output[:200]}...\n") # 打印前200个字符预览
            
            # --- 步骤 3: 调用评估模型进行打分 ---
            eval_prompt = construct_eval_prompt(eval_prompt_template, row, model_output)
            
            print("[发送评估打分请求]...")
            evaluation_result = eval_client.call_llm(eval_prompt)
            
            if evaluation_result:
                print(f"[评估结果]:\n{evaluation_result[0]}")
            else:
                print("[评估失败]: 未能获取评估结果。")

        print(f"\n{'='*20} 所有数据处理完毕 {'='*20}")
    else:
        print("程序因无法加载测试数据或Prompt模板而终止。")
