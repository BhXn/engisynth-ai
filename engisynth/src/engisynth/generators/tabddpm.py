"""
TabDDPM Adapter
基于去噪扩散概率模型 (DDPM) 的表格数据生成
特点：生成质量极高，特别适合复杂分布，但训练时间较长
注意：需要安装 tab-ddpm 库: pip install tab-ddpm
"""

from .base import BaseGenerator
import pandas as pd
import numpy as np
import torch
from typing import Optional, Dict, Any


class TabDDPMAdapter(BaseGenerator):
    """TabDDPM 模型适配器 - 基于扩散模型的表格数据生成器"""

    def __init__(self,
                 num_timesteps=1000,
                 gaussian_loss_type='mse',
                 scheduler='cosine',
                 model_type='mlp',
                 model_params=None,
                 num_epochs=1000,
                 batch_size=256,
                 learning_rate=1e-3,
                 device='auto',
                 seed=None,
                 **kwargs):
        """
        初始化 TabDDPM 模型

        Args:
            num_timesteps: 扩散步数
            gaussian_loss_type: 损失类型 ('mse' or 'kl')
            scheduler: 噪声调度器类型 ('linear' or 'cosine')
            model_type: 模型架构 ('mlp' or 'resnet')
            model_params: 模型参数字典
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            device: 计算设备 ('cpu', 'cuda', or 'auto')
            seed: 随机种子
        """
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.scheduler = scheduler
        self.model_type = model_type
        self.model_params = model_params or self._get_default_model_params()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = self._get_device(device)
        self.seed = seed
        self.kwargs = kwargs

        self.model = None
        self.preprocessor = None
        self.column_info = None

        if self.seed is not None:
            self._set_seed(self.seed)

    def _get_default_model_params(self) -> Dict[str, Any]:
        """获取默认模型参数"""
        return {
            'num_layers': 4,
            'hidden_dims': [256, 256, 256, 256],
            'dropout': 0.0,
            'activation': 'relu'
        }

    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def _preprocess_data(self, df: pd.DataFrame) -> torch.Tensor:
        """预处理数据：标准化数值特征，编码分类特征"""
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        processed_data = []
        self.column_info = {}

        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # 数值列：标准化
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[[col]])
                processed_data.append(scaled_data)
                self.column_info[col] = {
                    'type': 'numerical',
                    'scaler': scaler,
                    'dim': 1
                }
            else:
                # 分类列：One-hot 编码
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(df[col].fillna('missing'))
                n_classes = len(encoder.classes_)
                one_hot = np.eye(n_classes)[encoded]
                processed_data.append(one_hot)
                self.column_info[col] = {
                    'type': 'categorical',
                    'encoder': encoder,
                    'dim': n_classes
                }

        # 合并所有特征
        data_array = np.concatenate(processed_data, axis=1)
        return torch.FloatTensor(data_array).to(self.device)

    def _postprocess_data(self, generated_data: torch.Tensor, n_samples: int) -> pd.DataFrame:
        """后处理生成的数据：反标准化和解码"""
        generated_data = generated_data.cpu().numpy()

        result_df = pd.DataFrame()
        current_idx = 0

        for col, info in self.column_info.items():
            dim = info['dim']
            col_data = generated_data[:, current_idx:current_idx + dim]

            if info['type'] == 'numerical':
                # 反标准化数值列
                col_values = info['scaler'].inverse_transform(col_data)
                result_df[col] = col_values.flatten()
            else:
                # 解码分类列
                col_indices = np.argmax(col_data, axis=1)
                col_values = info['encoder'].inverse_transform(col_indices)
                result_df[col] = col_values

            current_idx += dim

        return result_df

    def fit(self, df: pd.DataFrame, **kwargs):
        """训练 TabDDPM 模型"""
        try:
            from tab_ddpm import GaussianMultinomialDiffusion
        except ImportError:
            raise ImportError(
                "TabDDPM requires the tab-ddpm library. "
                "Please install it using: pip install tab-ddpm"
            )

        # 预处理数据
        X_train = self._preprocess_data(df)

        # 计算输入维度
        input_dim = X_train.shape[1]

        # 创建扩散模型
        self.model = GaussianMultinomialDiffusion(
            num_classes=None,  # 连续数据
            denoise_fn=self._create_denoise_network(input_dim),
            num_timesteps=self.num_timesteps,
            gaussian_loss_type=self.gaussian_loss_type,
            scheduler=self.scheduler,
            device=self.device
        )

        # 训练模型
        self._train_model(X_train)

    def _create_denoise_network(self, input_dim: int):
        """创建去噪网络"""
        import torch.nn as nn

        class DenoiseNet(nn.Module):
            def __init__(self, input_dim, hidden_dims, activation='relu'):
                super().__init__()

                # 构建网络层
                layers = []
                prev_dim = input_dim

                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(self._get_activation(activation))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    if self.model_params.get('dropout', 0) > 0:
                        layers.append(nn.Dropout(self.model_params['dropout']))
                    prev_dim = hidden_dim

                # 输出层
                layers.append(nn.Linear(prev_dim, input_dim))

                self.network = nn.Sequential(*layers)

            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'tanh':
                    return nn.Tanh()
                elif activation == 'leaky_relu':
                    return nn.LeakyReLU()
                else:
                    return nn.ReLU()

            def forward(self, x, t):
                # t 是时间步嵌入，这里简化处理
                return self.network(x)

        return DenoiseNet(
            input_dim=input_dim,
            hidden_dims=self.model_params['hidden_dims'],
            activation=self.model_params.get('activation', 'relu')
        ).to(self.device)

    def _train_model(self, X_train: torch.Tensor):
        """训练扩散模型"""
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim

        # 创建数据加载器
        dataset = TensorDataset(X_train)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # 创建优化器
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # 训练循环
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]

                # 计算损失
                loss = self.model.get_loss(x)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        """生成合成数据"""
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # 生成数据
        self.model.eval()
        with torch.no_grad():
            generated_data = self.model.sample(n)

        # 后处理并返回 DataFrame
        return self._postprocess_data(generated_data, n)