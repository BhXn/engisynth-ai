"""
TVAE (Triplet-based Variational Autoencoder) Adapter
TVAE 使用变分自编码器架构，通过正则化的潜在空间学习数据分布
特点：训练速度快，生成质量稳定，适合中小规模数据集
"""

from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
from .base import BaseGenerator
import pandas as pd


class TVAEAdapter(BaseGenerator):
    """TVAE 模型适配器 - 基于变分自编码器的表格数据生成器"""

    def __init__(self,
                 epochs=300,
                 batch_size=500,
                 embedding_dim=128,
                 compress_dims=(128, 128),
                 decompress_dims=(128, 128),
                 l2_scale=1e-5,
                 loss_factor=2,
                 **kwargs):
        """
        初始化 TVAE 模型

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            embedding_dim: 嵌入维度
            compress_dims: 编码器隐藏层维度
            decompress_dims: 解码器隐藏层维度
            l2_scale: L2 正则化系数
            loss_factor: 损失函数权重因子
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2_scale = l2_scale
        self.loss_factor = loss_factor
        self.kwargs = kwargs
        self.model = None
        self.metadata = None

    def fit(self, df: pd.DataFrame, **kwargs):
        """训练 TVAE 模型"""
        # 检测数据元信息
        self.metadata = Metadata.detect_from_dataframe(df)

        # 创建 TVAE 模型
        self.model = TVAESynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            l2_scale=self.l2_scale,
            loss_factor=self.loss_factor,
            **self.kwargs
        )

        # 训练模型
        self.model.fit(df)

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        """生成合成数据"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return self.model.sample(num_rows=n)