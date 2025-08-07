import pandas as pd
from .base import Constraint


class RangeConstraint(Constraint):
    """范围约束：确保指定列的值在给定范围内"""
    def __init__(self, cols, min_val=None, max_val=None):
        # 支持单列或多列
        self.cols = cols if isinstance(cols, list) else [cols]
        self.min_val = min_val
        self.max_val = max_val

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """检查所有指定列是否都满足范围约束"""
        # 筛选出 DataFrame 中实际存在的列
        existing = [col for col in self.cols if col in df.columns]
        if not existing:
            # 没有列可检查时，视为全部满足
            return pd.Series(True, index=df.index)

        # 使用向量化比较，得到布尔 DataFrame
        mask_df = pd.DataFrame(True, index=df.index, columns=existing)
        if self.min_val is not None:
            mask_df &= df[existing].ge(self.min_val)
        if self.max_val is not None:
            mask_df &= df[existing].le(self.max_val)

        # 只有所有列都满足，才算该行满足约束
        return mask_df.all(axis=1)

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """将违反约束的值投影到约束范围内"""
        df_proj = df.copy()
        # 仅对存在的列进行裁剪
        existing = [col for col in self.cols if col in df_proj.columns]
        if existing:
            df_proj[existing] = df_proj[existing].clip(lower=self.min_val, upper=self.max_val)
        return df_proj