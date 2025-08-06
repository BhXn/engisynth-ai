import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .base import Constraint
from .sum_equal import SumEqual
from .range_constraint import RangeConstraint
from .ratio_constraint import RatioConstraint
from .monotonic_constraint import MonotonicConstraint
from .correlation_constraint import CorrelationConstraint


class ConstraintManager:
    """约束管理器：管理和应用多个约束"""

    def __init__(self, constraints_cfg: List[Dict[str, Any]] = None):
        self.constraints = []
        if constraints_cfg:
            self.load_constraints(constraints_cfg)

    def load_constraints(self, constraints_cfg: List[Dict[str, Any]]):
        """从配置加载约束"""
        for cfg in constraints_cfg:
            # 修复：根据传入对象类型决定访问方式
            if hasattr(cfg, 'type'):  # Pydantic 对象
                constraint_type = cfg.type
                get_attr = lambda attr, default=None: getattr(cfg, attr, default)
            else:  # 字典对象
                constraint_type = cfg.get('type')
                get_attr = lambda attr, default=None: cfg.get(attr, default)

            if constraint_type == 'sum_equal':
                constraint = SumEqual(
                    cols=get_attr('cols'),
                    value=get_attr('value', 1.0),
                    tol=get_attr('tol', 1e-4)
                )
            elif constraint_type == 'range':
                constraint = RangeConstraint(
                    cols=get_attr('cols'),
                    min_val=get_attr('min'),
                    max_val=get_attr('max')
                )
            elif constraint_type == 'ratio':
                cols = get_attr('cols')
                constraint = RatioConstraint(
                    col1=cols[0],
                    col2=cols[1],
                    ratio=get_attr('value'),
                    tol=get_attr('tol', 0.05)
                )
            elif constraint_type == 'monotonic':
                cols = get_attr('cols')
                constraint = MonotonicConstraint(
                    col_x=cols[0],
                    col_y=cols[1],
                    direction=get_attr('direction', 'increasing')
                )
            elif constraint_type == 'correlation':
                cols = get_attr('cols')
                constraint = CorrelationConstraint(
                    col1=cols[0],
                    col2=cols[1],
                    target_corr=get_attr('value'),
                    tol=get_attr('tol', 0.1)
                )
            else:
                continue

            self.constraints.append(constraint)

    def add_constraint(self, constraint: Constraint):
        """添加约束"""
        self.constraints.append(constraint)

    def project(self, df: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        """迭代投影数据到满足所有约束的空间"""
        df_proj = df.copy()

        for _ in range(max_iter):
            converged = True

            # 依次应用每个约束的投影
            for constraint in self.constraints:
                df_before = df_proj.copy()
                df_proj = constraint.project(df_proj)

                # 检查是否收敛
                if not df_proj.equals(df_before):
                    converged = False

            if converged:
                break

        return df_proj

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """评估所有约束的满足程度"""
        results = {}
        for i, constraint in enumerate(self.constraints):
            key = f"{constraint.__class__.__name__}_{i}"
            results[key] = constraint.evaluate(df)

        # 计算总体满足度
        if results:
            results['overall'] = np.mean(list(results.values()))
        else:
            results['overall'] = 1.0

        return results

    def filter_satisfied(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤出满足所有约束的行"""
        mask = pd.Series(True, index=df.index)

        for constraint in self.constraints:
            mask &= constraint.is_satisfied(df)

        return df[mask]