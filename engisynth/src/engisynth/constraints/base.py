from abc import ABC, abstractmethod
import pandas as pd


class Constraint(ABC):
    """约束基类"""

    @abstractmethod
    def is_satisfied(self, df: pd.DataFrame) -> pd.Series:
        """检查每行数据是否满足约束，返回布尔Series"""
        pass

    def project(self, df: pd.DataFrame) -> pd.DataFrame:
        """将数据投影到满足约束的空间（如果可能）"""
        return df

    def evaluate(self, df: pd.DataFrame) -> float:
        """评估约束的满足程度，返回0-1之间的分数"""
        satisfied = self.is_satisfied(df)
        return satisfied.sum() / len(satisfied) if len(satisfied) > 0 else 1.0