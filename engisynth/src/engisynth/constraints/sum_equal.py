import pandas as pd
import numpy as np
from .base import Constraint


class SumEqual(Constraint):
    """求和相等约束：确保指定列的和等于给定值"""

    def __init__(self, cols, value=1.0, tol=1e-4):
        self.cols = cols
        self.value = value
        self.tol = tol

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """检查每行的指定列之和是否等于目标值"""
        return (df[self.cols].sum(axis=1) - self.value).abs() < self.tol

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """将数据投影到满足约束的空间"""
        df_proj = df.copy()

        # 计算当前和
        current_sum = df_proj[self.cols].sum(axis=1)

        # 对于和不为0的行，按比例调整
        mask = current_sum != 0
        if mask.any():
            # 计算缩放因子
            scale_factor = self.value / current_sum[mask]

            # 按比例调整每列的值
            for col in self.cols:
                df_proj.loc[mask, col] *= scale_factor

        # 对于和为0的行，均匀分配
        zero_mask = ~mask
        if zero_mask.any():
            uniform_value = self.value / len(self.cols)
            for col in self.cols:
                df_proj.loc[zero_mask, col] = uniform_value

        return df_proj
