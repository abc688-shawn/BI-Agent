def evaluate_call_equivalence(gt_content: str, pred_content: str, judge_model: dict, row: dict = None) -> dict:
    """
    @param:
        gt_content: ground_truth
        pred_content: 模型返回结果
        judge_model: 用于评估的模型 会传递一个类似于下面的字典
        {"qwen3_moe_agent": {"addr": "123.181.192.99:6095", "model_name": "qwen3_32b"}}
    return:
        包含 is_match 键在内的 JSON 字典
    """
    gt_content = gt_content.split('</think>')[-1].strip()
    pred_content = pred_content.split('</think>')[-1].strip()
    # 1. 本地硬规则
    local_result = local_check(gt_content, pred_content)
    if local_result is not None:
        return local_result
    
    gt_query = extract_query(gt_content)
    pred_query = extract_query(pred_content)
    # 2. 结构 OK，交给 LLM 做语义一致性判定
    prompt = build_eval_prompt(gt_query, pred_query)
    while True:
        try:
            status_code, res = call_llm_for_agent(query=prompt,          model=judge_model['model_name'],
                                                  base_url=judge_model['addr'],
                                                  temperature=0, top_k=1, top_p=0.9)
    
            if status_code == 200:
                json_str = res.split('</think>')[-1].strip()
                if '```' in json_str:
                    json_str_list = JSON_BLOCK_RE.findall(json_str)
                    json_str = json_str_list[0]
                result = json.loads(json_str)
                break
            else:
                print('error for check answer, code != 200')
                continue
        except json.JSONDecodeError:
            print(f"模型输出不是合法 JSON：{json_str}")
            continue
        except Exception as e:
            print('check model error', e)
            continue
    
    return result
    ...