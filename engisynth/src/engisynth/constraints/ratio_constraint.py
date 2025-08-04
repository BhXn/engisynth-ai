import pandas as pd
import numpy as np
from .base import Constraint


class RatioConstraint(Constraint):
    """比例约束：确保两列之间保持特定的比例关系"""

    def __init__(self, col1, col2, ratio, tol=0.05):
        self.col1 = col1
        self.col2 = col2
        self.ratio = ratio
        self.tol = tol  # 容差，默认5%

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """检查两列的比例是否在容差范围内"""
        if self.col1 not in df.columns or self.col2 not in df.columns:
            return pd.Series(True, index=df.index)

        # 避免除零
        mask = df[self.col2] != 0
        actual_ratio = pd.Series(np.nan, index=df.index)
        actual_ratio[mask] = df.loc[mask, self.col1] / df.loc[mask, self.col2]

        # 检查比例是否在容差范围内
        lower_bound = self.ratio * (1 - self.tol)
        upper_bound = self.ratio * (1 + self.tol)

        return (actual_ratio >= lower_bound) & (actual_ratio <= upper_bound)