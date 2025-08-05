from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from .base import BaseGenerator
import pandas as pd


class CTGANAdapter(BaseGenerator):
    def __init__(self, epochs=300, batch_size=64, **kwargs):
        # SDV 的 CTGAN 参数
        self.epochs = epochs
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.model = None
        self.metadata = None

    def fit(self, df: pd.DataFrame, **kwargs):
        # 创建 metadata 对象 - SDV 需要这个来了解数据结构
        self.metadata = Metadata.detect_from_dataframe(df)

        # 创建 CTGAN 模型
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            pac=1,
            epochs=self.epochs,
            batch_size=self.batch_size,
            **self.kwargs
        )

        # 训练模型
        self.model.fit(df)

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # SDV 的 sample 方法直接返回 DataFrame
        return self.model.sample(num_rows=n)

