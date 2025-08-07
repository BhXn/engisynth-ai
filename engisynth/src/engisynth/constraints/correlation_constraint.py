import pandas as pd
from .base import Constraint


class CorrelationConstraint(Constraint):
    """相关性约束：确保两列之间的全局相关性在容差范围内"""

    def __init__(self, col1, col2, target_corr, tol=0.1):
        self.col1 = col1
        self.col2 = col2
        self.target_corr = target_corr
        self.tol = tol

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """全局约束：若相关性在容差内，则所有行均满足，否则均不满足。"""
        if not {self.col1, self.col2}.issubset(df.columns):
            return pd.Series(True, index=df.index)

        actual_corr = df[self.col1].corr(df[self.col2])
        satisfied = abs(actual_corr - self.target_corr) <= self.tol
        return pd.Series(satisfied, index=df.index)

    def evaluate(self, df: pd.DataFrame) -> float:
        """评估相关性满足程度，返回0-1之间的分数"""
        if not {self.col1, self.col2}.issubset(df.columns):
            return 1.0
        actual_corr = df[self.col1].corr(df[self.col2])
        diff = abs(actual_corr - self.target_corr)
        return max(0.0, 1.0 - diff / self.tol)

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """相关性投影暂不实现，默认返回原数据"""
        # 投影相关性需要全局重构，暂不支持自动投影
        return df