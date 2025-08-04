import pandas as pd
import numpy as np
from .base import Constraint


class RangeConstraint(Constraint):
    """范围约束：确保指定列的值在给定范围内"""

    def __init__(self, cols, min_val=None, max_val=None):
        self.cols = cols if isinstance(cols, list) else [cols]
        self.min_val = min_val
        self.max_val = max_val

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """检查所有指定列是否都满足范围约束"""
        mask = pd.Series(True, index=df.index)

        for col in self.cols:
            if col in df.columns:
                col_mask = pd.Series(True, index=df.index)
                if self.min_val is not None:
                    col_mask &= df[col] >= self.min_val
                if self.max_val is not None:
                    col_mask &= df[col] <= self.max_val
                mask &= col_mask

        return mask

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """将违反约束的值投影到约束范围内"""
        df_proj = df.copy()

        for col in self.cols:
            if col in df.columns:
                if self.min_val is not None:
                    df_proj[col] = df_proj[col].clip(lower=self.min_val)
                if self.max_val is not None:
                    df_proj[col] = df_proj[col].clip(upper=self.max_val)

        return df_proj