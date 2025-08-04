import pandas as pd
import numpy as np
from .base import Constraint


class CorrelationConstraint(Constraint):
    """相关性约束：确保列之间保持特定的相关性"""

    def __init__(self, col1, col2, target_corr, tol=0.1):
        self.col1 = col1
        self.col2 = col2
        self.target_corr = target_corr
        self.tol = tol

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """相关性是全局约束，返回全True"""
        return pd.Series(True, index=df.index)

    def evaluate(self, df: pd.DataFrame) -> float:
        """评估相关性约束的满足程度"""
        if self.col1 not in df.columns or self.col2 not in df.columns:
            return 1.0

        actual_corr = df[self.col1].corr(df[self.col2])
        diff = abs(actual_corr - self.target_corr)

        # 返回0-1之间的分数，完全满足时为1
        return max(0, 1 - diff / self.tol)