import pandas as pd
from .base import Constraint


class RatioConstraint(Constraint):
    """比例约束：确保两列之间保持特定的比例关系"""
    def __init__(self, col1, col2, ratio, tol=0.05):
        self.col1 = col1
        self.col2 = col2
        self.ratio = ratio
        self.tol = tol

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """检查两列的比例是否在容差范围内"""
        if not {self.col1, self.col2}.issubset(df.columns):
            return pd.Series(True, index=df.index)

        # 计算实际比例，自动处理除零产生的 inf
        actual_ratio = df[self.col1].div(df[self.col2])

        # 定义容差范围
        lower = self.ratio * (1 - self.tol)
        upper = self.ratio * (1 + self.tol)

        # 判断是否在范围内（NaN 和 inf 被视为不满足）
        return actual_ratio.between(lower, upper)

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """将数据投影到满足比例约束的空间：按比例调整 col1"""
        df_proj = df.copy()
        if not {self.col1, self.col2}.issubset(df_proj.columns):
            return df_proj

        # 对非零 col2 行，强制 col1 = col2 * ratio
        mask = df_proj[self.col2] != 0
        df_proj.loc[mask, self.col1] = df_proj.loc[mask, self.col2].mul(self.ratio)

        # 对 col2 为 0 的行，将 col1 设为 0
        df_proj.loc[~mask, self.col1] = 0

        return df_proj