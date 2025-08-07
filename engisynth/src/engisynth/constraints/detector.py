import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Any, Optional

class ConstraintDetector:
    """
    自动从数据集中探测潜在的约束条件。
    """
    def __init__(self, df: pd.DataFrame, threshold: float = 0.95, max_sum_cols: int = 4):
        """
        初始化探测器。

        Args:
            df (pd.DataFrame): 输入的数据集。
            threshold (float): 判定约束是否成立的行占比阈值。
            max_sum_cols (int): 检测 "sum_equal" 约束时，枚举的最大列数组合数量。
        """
        self.df = df
        self.threshold = threshold
        self.max_sum_cols = max_sum_cols
        self.numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def detect_range_constraints(self, quantile_low: float = 0.01, quantile_high: float = 0.99) -> List[Dict[str, Any]]:
        """探测范围约束 (range)。"""
        constraints = []
        for col in self.numerical_cols:
            # 使用分位数来避免极端离群点的影响，范围为1%和99%分位数
            low, high = self.df[col].quantile([quantile_low, quantile_high])
            if pd.notna(low) and pd.notna(high):
                constraints.append({
                    'type': 'range',
                    'cols': [col],
                    'min': round(float(low), 4),
                    'max': round(float(high), 4)
                })
        return constraints

    def detect_sum_equal_constraints(self, tol: float = 1e-4) -> List[Dict[str, Any]]:
        """探测和守恒约束 (sum_equal)。"""
        constraints = []
        # 限制列组合，避免计算量过大
        cols_to_check = [c for c in self.numerical_cols if self.df[c].nunique() > 1]
        
        for k in range(2, self.max_sum_cols + 1):
            for cols_tuple in combinations(cols_to_check, k):
                cols = list(cols_tuple)
                row_sum = self.df[cols].sum(axis=1)
                
                # 尝试找到一个最可能的常数值
                # 使用中位数对离群点更鲁棒
                constant_candidate = row_sum.median()
                
                satisfied_ratio = np.isclose(row_sum, constant_candidate, atol=tol).mean()
                
                if satisfied_ratio >= self.threshold:
                    constraints.append({
                        'type': 'sum_equal',
                        'cols': cols,
                        'value': round(float(constant_candidate), 4)
                    })
        return constraints

    def detect_category_allowed_constraints(self, max_categories: int = 20) -> List[Dict[str, Any]]:
        """探测离散合法值约束 (category_allowed)。"""
        constraints = []
        for col in self.categorical_cols:
            unique_values = self.df[col].unique()
            if 1 < len(unique_values) <= max_categories:
                constraints.append({
                    'type': 'category_allowed',
                    'cols': [col],
                    'values': [v for v in unique_values if pd.notna(v)]
                })
        return constraints

    def detect_all(self) -> List[Dict[str, Any]]:
        """运行所有探测器并返回合并后的约束列表。"""
        all_constraints = []
        print("Detecting range constraints...")
        all_constraints.extend(self.detect_range_constraints())
        print("Detecting sum_equal constraints...")
        all_constraints.extend(self.detect_sum_equal_constraints())
        print("Detecting category_allowed constraints...")
        all_constraints.extend(self.detect_category_allowed_constraints())
        return all_constraints

def auto_detect_feature_types(df: pd.DataFrame) -> Dict[str, str]:
    """基于启发式规则自动判断特征类型。"""
    feature_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # 启发式规则：如果数值列的唯一值数量很少，可能是一个编码后的类别
            if df[col].nunique() <= 20:
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'numerical'
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            feature_types[col] = 'categorical'
        elif pd.api.types.is_bool_dtype(df[col]):
            feature_types[col] = 'boolean'
    return feature_types
