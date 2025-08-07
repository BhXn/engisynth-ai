import pandas as pd
from sklearn.isotonic import IsotonicRegression
from .base import Constraint


class MonotonicConstraint(Constraint):
    """单调性约束：确保两列之间保持单调关系，支持检查与投影（使用等熵回归）"""

    def __init__(self, col_x, col_y, direction='increasing'):
        self.col_x = col_x
        self.col_y = col_y
        self.direction = direction  # 'increasing' or 'decreasing'

    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """
        标记不满足单调性的行：对排序后的相邻对进行检测，
        若出现违反单调性则将对应行标记为 False
        """
        # 确保列存在
        if not {self.col_x, self.col_y}.issubset(df.columns):
            return pd.Series(True, index=df.index)

        # 按 col_x 排序，计算相邻 diff
        sorted_df = df.sort_values(self.col_x)
        diffs = sorted_df[self.col_y].diff()

        # 根据方向判断违反
        if self.direction == 'increasing':
            viol_idx = diffs[diffs < 0].index
        else:
            viol_idx = diffs[diffs > 0].index

        # 生成标记，默认 True，违反的标 False
        mask = pd.Series(True, index=df.index)
        mask.loc[viol_idx] = False
        return mask

    def evaluate(self, df: pd.DataFrame) -> float:
        """
        返回 [0,1] 分数：1 表示完全单调，=1 - (违反对数/总对数)
        """
        if not {self.col_x, self.col_y}.issubset(df.columns):
            return 1.0

        sorted_df = df.sort_values(self.col_x)
        diffs = sorted_df[self.col_y].diff().dropna()

        if diffs.empty:
            return 1.0

        if self.direction == 'increasing':
            violations = (diffs < 0).sum()
        else:
            violations = (diffs > 0).sum()

        return 1.0 - violations / len(diffs)

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用等熵回归（Isotonic Regression）对 y 进行投影，
        强制其在 x 排序下保持单调性
        """
        df_proj = df.copy()
        if not {self.col_x, self.col_y}.issubset(df_proj.columns):
            return df_proj

        # 获取排序索引，将 x, y 提取
        sorted_idx = df_proj[self.col_x].argsort()
        x = df_proj.loc[sorted_idx, self.col_x].values
        y = df_proj.loc[sorted_idx, self.col_y].values

        # 拟合并投影
        ir = IsotonicRegression(increasing=(self.direction == 'increasing'))
        y_proj = ir.fit_transform(x, y)

        # 写回投影结果
        df_proj.loc[sorted_idx, self.col_y] = y_proj
        return df_proj
