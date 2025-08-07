import pandas as pd
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

        # 1. 计算每行当前和
        sums = df_proj[self.cols].sum(axis=1)

        # 2. 选出未满足约束的行
        mask_unsat = (sums - self.value).abs() >= self.tol
        if not mask_unsat.any():
            return df_proj

        # 3. 对于和不为零的未满足行，按比例缩放
        nonzero = (sums != 0) & mask_unsat
        if nonzero.any():
            factors = self.value / sums.loc[nonzero]
            df_proj.loc[nonzero, self.cols] = (
                df_proj.loc[nonzero, self.cols]
                .multiply(factors, axis=0)
            )

        # 4. 对于和为零的未满足行，均匀分配
        zero_sum = (~nonzero) & mask_unsat
        if zero_sum.any():
            uniform_value = self.value / len(self.cols)
            df_proj.loc[zero_sum, self.cols] = uniform_value

        return df_proj
