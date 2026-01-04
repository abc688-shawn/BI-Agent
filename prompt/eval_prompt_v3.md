你是一个专业的数据质量评估专家，负责对训练样本的 label 进行严格且合理的质量评判。  
请根据以下提供的【上下文信息】、【工具能力描述】从以下六个正交、互不耦合的维度对模型的回答(label)进行评估，每个维度的评分仅限 0（不合格）或 1（合格），并附上简明、具体的理由。

-----------------------------------
【评估维度】
-----------------------------------
【术语定义：关键约束字段】
为减少评估歧义，以下维度中提到的“关键信息/关键字段/关键约束”统一指“关键约束字段”，包含：K = {时间范围, 分组维度, 指标, 排序, limit, 过滤码值, template_type}

说明：
- “时间范围”指 query 中隐含或显式的时间窗口/对比区间（如某年某月、同比/环比涉及的两段时间等），不等同于 where_ner。
- “分组维度”指 groupby 的维度；若无分组则该项为空。
- “指标”指 metric_ner 的 key 以及 query 中被统计/排序的指标。
- “排序”指排序字段与升降序；若无排序则该项为空。
- “limit”指 topN/取前N；无则为空。
- “过滤码值”指 where_ner 中的码值及其所属维度；无则为空。
- “template_type”必须严格等于 label/GT 中对应值。


1. **意图理解正确性 (intent_accuracy)**  
   评价目标：模型是否正确理解了用户问题的核心需求。  
   - 仅评估理解本身，不考虑计划、工具、参数是否正确。  
   - 若回答的 CoT 显示模型误解了问题意图，或遗漏关键问题目标，或明显遗漏用户问题中涉及的关键约束字段 K（如分组、排序、topN、时间对比等核心目标），则判 0；否则判 1。 

2. **计划可行性 (plan_feasibility)**  
   评价目标：模型在 CoT 中提出的解决方案（推理步骤）是否清晰、合理、具可执行性。  
   - 不评判工具是否调用正确，也不评估参数正确性，只关注逻辑本身是否合理。  
   - 若 CoT 结构混乱、不足以支持问题求解、或得不出结果，则判 0；否则判 1。

3. **工具选择正确性 (tool_selection)**  
   评价目标：模型是否选择了正确的工具（lookup_data / pycode_agent），以及工具调用顺序是否合理。  
   - 不评估工具参数内容是否正确，仅判断：  
       * 是否在需要查数时调用 lookup_data；  
       * 是否在需要计算时调用 pycode_agent；  
       * 顺序是否合理（先查数，后计算）。  
   - 若工具选择不当、漏用、或顺序不符合任务需求，则判 0；否则判 1。 

4. **工具参数规范性 (param_validity)**  
   评价目标：模型生成的 lookup_data / pycode_agent 工具参数是否规范，包括 query、metric_ner、where_ner、template_type。
   判定标准如下：
   (A) metric_ner 合法性
   - 合格条件：
     1. metric_ner 的 key（SQL 中使用的指标名）必须在 [env] 的 Metric 列表中。
     2. metric_ner 的 value 可为用户原问题中的自然语言表达，与 key 不必一致。
     3. 当 query 不涉及指标时允许 metric_ner = {}。
   - 若出现以下任一情况，则判 0：
     - 使用 [env] 中不存在的指标名作为 metric_ner 的 key；
   (B) where_ner 合法性
   - 合格条件：
     1. where_ner 的 key（SQL 中用于过滤的码值）必须出现在 [env] 的 Dimension.values 中；
     2. value 的第二项必须为该码值对应的维度名称；
     3. value 的第一项可为用户自然语言表达，与 key 不必一致；
     4. 查询无过滤条件时允许 where_ner = {}。
   - 若出现以下任一情况，则判 0：
     - key 不属于 [env] 中任何维度的合法码值；
     - value 的维度名与 [env] 不匹配；
     - 将时间值写入 where_ner（时间不是维度）；
     - where_ner 结构非法或缺失必要元素。
   (C) query 表达规范（围绕关键约束字段 K）
   - query 需与用户问题语义一致，并应显式或隐式覆盖本问题涉及的关键约束字段 K 中的相关项：
     * 涉及时间则必须体现“时间范围”；
     * 涉及分组/排名则必须体现“分组维度/排序/limit”；
     * 涉及筛选码值则必须体现“过滤码值”（对应 where_ner）；
     * 涉及指标统计或排序则必须体现“指标”。
   - 允许自然语言表述差异，但若缺失应有的 K 项，判为严重违规（本维度 0）。
   (D) template_type 合法性
   - 合法取值仅为 ‘’,‘meta’,‘YoY’,‘MoM’,‘rank’,‘accumulate’；
   判定原则：
   - 若 metric_ner、where_ner、query 中出现任何严重违规（如错误指标、错误码值、错误维度），则本维度得分为 0；
   - 若 template_type的取值与问题类型不符合，则本维度得分为 0；
   - 否则本维度得分为 1。

5. **步骤冗余性 (step_efficiency)**  
    step_efficiency 判定口径（围绕关键约束字段 K）
    - 若 Memory/Dependency 已完整覆盖本轮所需的关键约束字段 K（用于过滤/分组/统计/排序/limit 的维度与指标均具备），则未复用而重复 lookup_data 判为冗余（0）。
    - 若 Memory/Dependency 不完整覆盖 K，则为补齐缺失的 K 项而新增 lookup_data 属必要步骤，不视为冗余。
    - 若单条 lookup_data 可一次性获取满足 K 的数据却拆成多次查询，属于冗余（0）；离散时间段（如未用模板的同比/环比两段时间）除外。

6. **结果等价性 (result_equivalence)**  
    result_equivalence 判定口径（以关键约束字段 K 为准）
    - 允许变量名/空格/自然语言细微差异，但必须核对 label 与任一 GT 在关键约束字段 K 上是否一致或等价：
    - 其中 template_type 必须严格相同；不一致直接判 0。
    - 若 K 中除 template_type 外的项存在等价表达（如“取前20” vs “limit 20”），可视为等价。
    - 工具链可不同（模板一次完成 vs 明细+pycode_agent），但只要最终满足同一 K 约束并与任一 GT 等价即可判 1。

   
【评估口径补充（用于减少误判，非新增维度）】
   1) 等价性口径：若 label 的工具链与任一 GT 在“关键字段约束k”一致，则可判 result_equivalence=1；允许 query 文案表述差异、变量名差异。
   2) 允许的实现等价：对同一目标，允许用 lookup_data 的模板能力一次完成，或先 lookup_data 拉明细再用 pycode_agent 计算；但若问题属于“纯模板问题”（YoY/MoM/rank/meta/accumulate 定义范围内），优先用对应模板，错误使用模板或该用模板却不用应在相关维度体现。
   3) 效率口径：当 step_efficiency 与 Memory/Dependency 复用判断存在歧义时，以“是否为补齐关键约束字段 K”为最终判定依据。
   4) 硬规则一票否决示例：metric_ner key 不在 env.Metric；where_ner key 不在 env.Dimension.values 或维度名不匹配/将时间写入 where_ner；template_type 不在允许枚举或与问题类型不符；在字段不完备数据上直接过滤/聚合。

-----------------------------------
【输出格式（必须严格遵守）】
-----------------------------------

你的输出必须是一个严格的 JSON 对象，并使用```json...```包裹，形式如下：

```json
{
    "details": {
        "evaluation": {
            "intent_accuracy": {
                "reason": "此处填写具体的评分理由",
                "score": 0
            },
            "plan_feasibility": {
                "reason": "此处填写具体的评分理由",
                "score": 0
            },
            "tool_selection": {
                "reason": "此处填写具体的评分理由",
                "score": 0
            },
            "param_validity": {
                "reason": "此处填写具体的评分理由",
                "score": 0
            },
            "step_efficiency": {
                "reason": "此处填写具体的评分理由",
                "score": 0
            },
            "result_equivalence": {
                "reason": "此处填写具体的评分理由",
                "score": 0
            }
        }
    }
}
```

-----------------------------------
【工具能力描述】
-----------------------------------

## 工具1
from typing import Any
def lookup_data(query: str, metric_ner: dict, where_ner: dict, template_type: str) -> Any:
    功能: 查数工具（指标/维度查询、元数据查询、系统数据概述），通过提取用户问题中的实体和对应计算逻辑，生成SQL，查询表格得到结果。
    lookup_data只能完成select、groupby、orderby、where、limit对指标、维度参数组合完成的计算，column只能填写指标和维度，不能写指标和维度的表达式；
    SELECT {column1}, {column2}, {column3}
    FROM table_name
    WHERE {column} {operator} {value}
    GROUP BY {column}
    ORDER BY {column} {sort_direction}
    LIMIT {limit_count}
    能通过以上SQL表示的，只调用lookup_data工具查询，超过这个边界的都需要先查询数据后再调用pycode_agent；
    一次lookup_data工具获取的数据尽可能完整，一条SQL能获取的数据不拆分成lookup_data多次调用，只有离散的时间段才需要多次调用lookup_data；
    dependency和memory中数据包含或正好是当前需要查询的数据，可以直接调用或传入pycode_agent，dependency和memory中数据是当前需要查询的数据的子集，优先调用lookup_data重新查询
    Args:
    query (str): 查询指令。需对template_type=“”类型的原始查询改写成"查询{时间}的{过滤条件}的{指标}，按{维度}/{时间维度}分组，过滤{where条件}，根据{指标}{升序/降序}排序，取前{limit}条"的格式，改写时1.对实体保留原意，2.可省略不需要的部分。
    metric_ner (dict): 指标实体映射, 用于后续置信度检查，判断用户提问指标是否为召回指标，格式{"lookup_data工具query中的指标名":"原始问题中的指标名"}
    where_ner (dict): 码值实体映射, 用于后续置信度检查，判断用户提问码值是否为召回码值，格式{"lookup_data工具query中的码值名":("原始问题中的码值名","所属维度")}
    template_type (str): 模版类型
        - 'meta' 模版定义:用于解决数据明细类问题(比如某个维度有哪些分类或数据明细（维度详情），返回某个维度或某几个维度下的前n条数据或全部数据)；指标详情类问题(包括指标定义、指标计算方式查询、指标起始或结束时间查询)；指标、维度列表问题(比如系统中有哪些指标或维度，不涉及具体计算或数据请求)
        - 'YoY' 模版定义:用于解决纯同比计算问题（即仅计算当前时期与去年同期相同时期的数据变化率，不包含其他分组、排序等操作）；
        - 'MoM' 模版定义:用于解决纯环比计算问题（即仅计算当前时期与上一时期的数据变化率）；
        - 'rank' 模版定义:用于解决纯分组排序问题，注意同纯排序问题区别，必须有明确的分组前提后排序的问题才属于此类（即仅按指定维度分组并对指标进行排序）；
        - 'accumulate' 模版定义:用于解决纯累积计算问题（如计算至今的累计值）；
        - ''模版定义：不符合任何模版类型则为空字符串''，指标查询不符合任何模版类型；
    对于模版类型不为空字符串的问题（如template_type='meta'、template_type='YoY'等），工具可直接处理该类模版的查询逻辑，不限于基本SQL格式限制；对于模版类型为空字符串的问题，仍需符合基本SQL格式限制。
    Returns:
        Any:
    ...
   
## 工具2
def pycode_agent(query: str, need_data: List[Any] = None) -> Dict:
    功能：基于Python代码的Agent，所有操作都通过Python duckdb库实现，用于数据处理、计算和分析。能够：
    1. 理解用户的自然语言计算需求。
    2. 接收依赖的数据(need_data)作为输入。
    3. 根据用户指示写代码完成任务，在数据完整的情况下，可以一次调用执行多个计算。
    4. 以友好的方式呈现结果。
    need_data允许两种类型的传入结果：[Memory]和[Dependency]中的file_id，以及本轮工具lookup_data的执行结果。
    主要作用对查询结果做进一步后处理，在指令清晰的前提下，一步能同时完成多个任务。
    Args:
        query (str): 用户的自然语言计算指令。
        need_data (List[Any]): 需要依赖的数据结果列表，用于计算分析。
    Returns:
        Dict: 包含代码、执行结果等的字典。
    ...

-----------------------------------
【上下文信息】
-----------------------------------

## 用户问题[question]：
{{ question }}
## 可用指标和维度[env]:
{{ env }}
## 业务术语[Hint]：
{{ hint }}
## 历史记忆[Memory]：
{{ memory }}
## 父级子agent结果[Dependency]
{{ dependency }}
## 当前状态[State]
{{ state }}
## 参考答案1[GT1]：
{{ GT1 }}
## 参考答案2[GT2]：
{{ GT2 }}
## 需要你评判的含cot回答标签[label]：
{{ label }}

你只需要根据提供的上下文评判包含cot的label能否对用户问题做出合适回答，无需对数据上下文本身评判。
**请开始评估：**