# 版本说明
## bi_prompt
- v1：最初版本的bi提示词
- v2：在最初版本bi提示词基础上进行了格式等方面的微调，且增加了join问题的描述（效果不太理想）
- v3：为了roll out得到指标模版训练样本将fewshot增加至5个
- v4：在最初版本的bi提示词的基础上新增对template_type的描述
## eval_prompt
- v1：最初版本评估提示词
- v2：带有两个评估fewshot的新版本评估提示词
- v3：去掉了之前添加的两个评估fewshot，新增额外template_type描述的评估提示词
## meta_eval_prompt_v1.md
- 用于给评估器的评估进行打分
## test