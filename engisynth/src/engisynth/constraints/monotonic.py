import pandas as pd
import numpy as np
from .base import Constraint


class MonotonicConstraint(Constraint):
    """单调性约束：确保两列之间保持单调关系"""

    def __init__(self, col_x, col_y, direction='increasing'):
        self.col_x = col_x
        self.col_y = col_y
        self.direction = direction  # 'increasing' or 'decreasing'

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """检查单调性约束（这是一个软约束，难以逐行检查）"""
        # 单调性是全局约束，返回全True，实际检查在evaluate方法中
        return pd.Series(True, index=df.index)

    def evaluate(self, df: pd.DataFrame) -> float:
        """评估单调性程度，返回0-1之间的分数"""
        if self.col_x not in df.columns or self.col_y not in df.columns:
            return 1.0

        # 按x列排序
        sorted_df = df.sort_values(by=self.col_x)
        y_values = sorted_df[self.col_y].values

        # 计算单调性违反的比例
        if self.direction == 'increasing':
            violations = np.sum(np.diff(y_values) < 0)
        else:
            violations = np.sum(np.diff(y_values) > 0)

        total_pairs = len(y_values) - 1
        if total_pairs == 0:
            return 1.0

        return 1.0 - (violations / total_pairs)