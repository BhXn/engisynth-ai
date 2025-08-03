from synthcity.plugins import Plugins
from .base import BaseGenerator
import pandas as pd

class CTGANAdapter(BaseGenerator):
    def __init__(self, epochs=300, batch_size=64, **kwargs):
        # SynthCity 的 CTGAN 插件用 n_iter 表示训练轮数
        self.plugin = Plugins().get("ctgan", n_iter=epochs, batch_size=batch_size, **kwargs)

    def fit(self, df: pd.DataFrame, **kwargs):
        # 直接传 DataFrame；如果有条件可以用 cond=...
        self.plugin.fit(df, **kwargs)

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        # generate 返回的是 DataLoader，要调用 .dataframe()
        loader = self.plugin.generate(count=n, **kwargs)
        return loader.dataframe()

