from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from .base import BaseGenerator
import pandas as pd


class CTGANAdapter(BaseGenerator):
    def __init__(self, **kwargs):
        self.model_kwargs = kwargs
        self.model = None
        self.metadata = None

    def fit(self, df: pd.DataFrame, **kwargs):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(df)
        
        # 从kwargs中移除CTGANSynthesizer不接受的参数
        init_kwargs = self.model_kwargs.copy()
        init_kwargs.pop('max_rejection_samples', None)

        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            **init_kwargs
        )
        self.model.fit(df)

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # SDV 的 sample 方法直接返回 DataFrame
        return self.model.sample(num_rows=n)

