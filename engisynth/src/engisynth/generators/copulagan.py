"""
CopulaGAN Adapter
结合了 Copula 统计方法和 GAN 的优势
特点：能够很好地捕捉变量间的依赖关系，适合具有复杂相关性的数据
"""

from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import Metadata
from .base import BaseGenerator
import pandas as pd


class CopulaGANAdapter(BaseGenerator):
    """CopulaGAN 模型适配器 - 结合 Copula 和 GAN 的表格数据生成器"""

    def __init__(self,
                 epochs=300,
                 batch_size=500,
                 generator_dim=(256, 256),
                 discriminator_dim=(256, 256),
                 generator_lr=2e-4,
                 discriminator_lr=2e-4,
                 discriminator_steps=1,
                 log_frequency=True,
                 verbose=False,
                 **kwargs):
        """
        初始化 CopulaGAN 模型

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            generator_dim: 生成器隐藏层维度
            discriminator_dim: 判别器隐藏层维度
            generator_lr: 生成器学习率
            discriminator_lr: 判别器学习率
            discriminator_steps: 每个生成器步骤对应的判别器训练步数
            log_frequency: 是否对数值列使用对数频率编码
            verbose: 是否显示训练进度
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.kwargs = kwargs
        self.model = None
        self.metadata = None

    def fit(self, df: pd.DataFrame, **kwargs):
        """训练 CopulaGAN 模型"""
        # 检测数据元信息
        self.metadata = Metadata.detect_from_dataframe(df)

        # 创建 CopulaGAN 模型
        self.model = CopulaGANSynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            discriminator_steps=self.discriminator_steps,
            log_frequency=self.log_frequency,
            verbose=self.verbose,
            **self.kwargs
        )

        # 训练模型
        self.model.fit(df)

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        """生成合成数据"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return self.model.sample(num_rows=n)