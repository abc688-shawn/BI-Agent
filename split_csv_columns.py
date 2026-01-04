import re
import pandas as pd


def _extract(text: str, pattern: str) -> str:
    if not isinstance(text, str):
        return ""
    # 关键：统一换行符，避免 \r\n 导致 lookahead 失败
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if not text.strip():
        return ""

    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        return ""

    s = m.group(1)
    if s is None:
        return ""

    s = s.strip()
    s = "\n".join(line.rstrip() for line in s.splitlines()).strip()
    return s


def split_all_column(df: pd.DataFrame, all_col: str = "all") -> pd.DataFrame:
    """
    从 df[all_col] 中抽取：
      - memory: ## 前轮对话记忆[Memory] ... ## 召回上下文信息
      - env:    ### [Metric]: ... （到 ### 业务术语[Hint] 或 ## 父级子agent结果[Dependency] 之前）
      - hint:   '[思考逻辑规则hint]': ... （到 ## 父级子agent结果[Dependency] 之前）
      - dependency: ## 父级子agent结果[Dependency] ... （到 ## 当前状态[State] 之前）
      - state:  ## 当前状态[State] ... （到文本末尾）
    """
    if all_col not in df.columns:
        raise ValueError(f"找不到列：{all_col}. 当前列：{list(df.columns)}")

    memory_pat = r"##\s*前轮对话记忆\[Memory\]\s*(?:\n)+([\s\S]*?)(?=(?:\n)+##\s*召回上下文信息|$)"


    env_pat = r"(###\s*\[Metric\]:.*?)(?=\n###\s*业务术语\[Hint\]|\n##\s*父级子agent结果\[Dependency\]|$)"

    hint_pat = r"(\'\[思考逻辑规则hint\]\':.*?)(?=\n##\s*父级子agent结果\[Dependency\]|$)"

    dependency_pat = r"##\s*父级子agent结果\[Dependency\]\s*\n(.*?)(?=\n##\s*当前状态\[State\]|$)"

    state_pat = r"##\s*当前状态\[State\]\s*\n(.*)$"


    df = df.copy()
    df["memory"] = df[all_col].apply(lambda x: _extract(x, memory_pat))
    df["env"] = df[all_col].apply(lambda x: _extract(x, env_pat))
    df["hint"] = df[all_col].apply(lambda x: _extract(x, hint_pat))
    df["dependency"] = df[all_col].apply(lambda x: _extract(x, dependency_pat))
    df["state"] = df[all_col].apply(lambda x: _extract(x, state_pat))
    return df


if __name__ == "__main__":
    # 读入你的 CSV（把路径改成你的真实路径）
    input_csv = "data/1222.csv"
    output_csv = "output/output_1222.csv"

    df = pd.read_csv(input_csv, encoding="utf-8")
    df2 = split_all_column(df, all_col="all")

    df2.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已输出：{output_csv}")
