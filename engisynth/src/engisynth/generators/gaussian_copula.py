"""
GaussianCopula Adapter
基于高斯 Copula 的统计方法
特点：训练速度极快，数学基础扎实，适合快速原型开发和基准对比
"""

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from .base import BaseGenerator
import pandas as pd


class GaussianCopulaAdapter(BaseGenerator):
    """GaussianCopula 模型适配器 - 基于统计方法的表格数据生成器"""

    def __init__(self,
                 numerical_distributions=None,
                 default_distribution='beta',
                 min_value_clip=None,
                 max_value_clip=None,
                 **kwargs):
        """
        初始化 GaussianCopula 模型

        Args:
            numerical_distributions: 数值列的分布类型字典
                可选: 'norm', 'beta', 'truncnorm', 'uniform', 'gamma', 'gaussian_kde'
            default_distribution: 默认分布类型
            min_value_clip: 数值的最小值裁剪
            max_value_clip: 数值的最大值裁剪
        """
        self.numerical_distributions = numerical_distributions or {}
        self.default_distribution = default_distribution
        self.min_value_clip = min_value_clip
        self.max_value_clip = max_value_clip
        self.kwargs = kwargs
        self.model = None
        self.metadata = None

    def fit(self, df: pd.DataFrame, **kwargs):
        """训练 GaussianCopula 模型"""
        # 检测数据元信息
        self.metadata = Metadata.detect_from_dataframe(df)

        # 创建 GaussianCopula 模型
        self.model = GaussianCopulaSynthesizer(
            metadata=self.metadata,
            numerical_distributions=self.numerical_distributions,
            default_distribution=self.default_distribution,
            **self.kwargs
        )

        # 设置数值裁剪
        if self.min_value_clip is not None or self.max_value_clip is not None:
            constraints = []
            for col in df.select_dtypes(include=['number']).columns:
                constraint = {
                    'constraint_class': 'ScalarRange',
                    'constraint_parameters': {
                        'column_name': col,
                        'low_value': self.min_value_clip,
                        'high_value': self.max_value_clip,
                        'strict_boundaries': False
                    }
                }
                constraints.append(constraint)
            if constraints:
                self.model.add_constraints(constraints)

        # 训练模型
        self.model.fit(df)

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        """生成合成数据"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return self.model.sample(num_rows=n)