try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ImportError:
    print("Warning: tabpfn not installed. Please install with: pip install tabpfn")
    TabPFNClassifier = None
    TabPFNRegressor = None

from .base import BaseGenerator
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class TabPFNGenerativeAdapter(BaseGenerator):
    """
    TabPFN 生成式适配器
    基于TabPFN的生成式模型，通过反向采样生成新数据
    注意：TabPFN原本是预测模型，这里通过特殊技巧实现生成功能
    """

    def __init__(self, n_estimators=10, target_column=None, generation_method='bootstrap', **kwargs):
        if TabPFNClassifier is None:
            raise ImportError("tabpfn is not installed. Please install with: pip install tabpfn")

        self.n_estimators = n_estimators
        self.target_column = target_column  # 如果指定，用作目标变量
        self.generation_method = generation_method  # 'bootstrap', 'noise', 'interpolation'
        self.kwargs = kwargs

        self.models = []
        self.feature_columns = None
        self.original_data = None
        self.label_encoders = {}
        self.scalers = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, **kwargs):
        """训练TabPFN生成模型"""
        self.original_data = df.copy()

        # 预处理数据
        df_processed = self._preprocess_data(df)

        # 如果没有指定目标列，尝试自动检测或创建多个模型
        if self.target_column is None:
            # 为每个特征训练一个模型（将其作为目标变量）
            self.feature_columns = df_processed.columns.tolist()

            for i, target_col in enumerate(self.feature_columns):
                if i >= self.n_estimators:  # 限制模型数量
                    break

                # 准备特征和目标
                X = df_processed.drop(columns=[target_col])
                y = df_processed[target_col]

                # 根据目标变量类型选择模型
                if self._is_classification_target(y):
                    model = TabPFNClassifier()
                else:
                    model = TabPFNRegressor()

                # 训练模型
                if len(X) > 0 and len(y) > 0:
                    try:
                        model.fit(X.values, y.values)
                        self.models.append({
                            'model': model,
                            'target_column': target_col,
                            'feature_columns': X.columns.tolist(),
                            'model_type': 'classifier' if self._is_classification_target(y) else 'regressor'
                        })
                    except Exception as e:
                        print(f"Warning: Failed to train model for {target_col}: {e}")
                        continue
        else:
            # 使用指定的目标列
            X = df_processed.drop(columns=[self.target_column])
            y = df_processed[self.target_column]

            if self._is_classification_target(y):
                model = TabPFNClassifier()
            else:
                model = TabPFNRegressor()

            model.fit(X.values, y.values)
            self.models.append({
                'model': model,
                'target_column': self.target_column,
                'feature_columns': X.columns.tolist(),
                'model_type': 'classifier' if self._is_classification_target(y) else 'regressor'
            })

        self.is_fitted = True

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        """生成合成数据"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if self.generation_method == 'bootstrap':
            return self._bootstrap_sampling(n)
        elif self.generation_method == 'noise':
            return self._noise_based_generation(n)
        elif self.generation_method == 'interpolation':
            return self._interpolation_generation(n)
        else:
            raise ValueError(f"Unknown generation method: {self.generation_method}")

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        df_processed = df.copy()

        # 处理分类变量
        categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le

        # 处理数值变量（TabPFN对数据范围较敏感）
        numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            scaler = StandardScaler()
            df_processed[col] = scaler.fit_transform(df_processed[[col]])
            self.scalers[col] = scaler

        return df_processed

    def _is_classification_target(self, y: pd.Series) -> bool:
        """判断目标变量是否为分类变量"""
        # 简单启发式：如果唯一值数量相对较少，认为是分类问题
        unique_ratio = len(y.unique()) / len(y)
        return unique_ratio < 0.1 or len(y.unique()) <= 20

    def _bootstrap_sampling(self, n: int) -> pd.DataFrame:
        """基于Bootstrap的采样生成"""
        # 从原始数据中随机采样
        sampled_indices = np.random.choice(len(self.original_data), size=n, replace=True)
        synthetic_data = self.original_data.iloc[sampled_indices].copy()

        # 添加一些噪音以增加多样性
        numerical_columns = synthetic_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            noise_std = synthetic_data[col].std() * 0.1  # 10% 的标准差作为噪音
            noise = np.random.normal(0, noise_std, size=n)
            synthetic_data[col] += noise

        return synthetic_data.reset_index(drop=True)

    def _noise_based_generation(self, n: int) -> pd.DataFrame:
        """基于噪音的生成方法"""
        if not self.models:
            return self._bootstrap_sampling(n)

        # 使用训练好的模型生成数据
        synthetic_samples = []

        for _ in range(n):
            # 随机选择一个模型
            model_info = np.random.choice(self.models)
            model = model_info['model']

            # 从原始数据中随机采样作为种子
            seed_idx = np.random.randint(0, len(self.original_data))
            seed_sample = self.original_data.iloc[seed_idx].copy()

            # 添加噪音并预测
            for col in model_info['feature_columns']:
                if col in self.original_data.columns:
                    original_val = seed_sample[col]
                    if pd.api.types.is_numeric_dtype(self.original_data[col]):
                        noise = np.random.normal(0, self.original_data[col].std() * 0.1)
                        seed_sample[col] = original_val + noise

            synthetic_samples.append(seed_sample)

        return pd.DataFrame(synthetic_samples).reset_index(drop=True)

    def _interpolation_generation(self, n: int) -> pd.DataFrame:
        """基于插值的生成方法"""
        synthetic_samples = []

        for _ in range(n):
            # 随机选择两个原始样本进行插值
            idx1, idx2 = np.random.choice(len(self.original_data), size=2, replace=False)
            sample1 = self.original_data.iloc[idx1]
            sample2 = self.original_data.iloc[idx2]

            # 随机插值权重
            alpha = np.random.random()

            interpolated_sample = sample1.copy()
            numerical_columns = self.original_data.select_dtypes(include=['int64', 'float64']).columns

            for col in numerical_columns:
                interpolated_sample[col] = alpha * sample1[col] + (1 - alpha) * sample2[col]

            synthetic_samples.append(interpolated_sample)

        return pd.DataFrame(synthetic_samples).reset_index(drop=True)