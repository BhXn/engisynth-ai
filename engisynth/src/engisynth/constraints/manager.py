import pandas as pd
from typing import List, Dict, Any

from .base import Constraint
from .sum_equal import SumEqual
from .range_constraint import RangeConstraint
from .ratio_constraint import RatioConstraint
from .monotonic_constraint import MonotonicConstraint
from .correlation_constraint import CorrelationConstraint


# 字符串类型到约束类的工厂映射
_CONSTRAINT_FACTORIES: Dict[str, Any] = {
    'sum_equal': lambda get: SumEqual(
        cols=get('cols'),
        value=get('value', 1.0),
        tol=get('tol', 1e-4)
    ),
    'range': lambda get: RangeConstraint(
        cols=get('cols'),
        min_val=get('min'),
        max_val=get('max')
    ),
    'ratio': lambda get: RatioConstraint(
        col1=get('cols')[0],
        col2=get('cols')[1],
        ratio=get('value'),
        tol=get('tol', 0.05)
    ),
    'monotonic': lambda get: MonotonicConstraint(
        col_x=get('cols')[0],
        col_y=get('cols')[1],
        direction=get('direction', 'increasing')
    ),
    'correlation': lambda get: CorrelationConstraint(
        col1=get('cols')[0],
        col2=get('cols')[1],
        target_corr=get('value'),
        tol=get('tol', 0.1)
    ),
}


class ConstraintManager:
    """约束管理器：加载、评估、投影多个约束"""

    def __init__(self, constraints_cfg: List[Any] = None):
        self.constraints: List[Constraint] = []
        if constraints_cfg:
            self.load_constraints(constraints_cfg)

    def load_constraints(self, constraints_cfg: List[Any]) -> None:
        """根据配置加载约束，支持 dict 或带属性的对象"""
        for cfg in constraints_cfg:
            # 统一获取类型和参数
            if hasattr(cfg, 'type'):
                ctype = getattr(cfg, 'type', None)
                get = lambda attr, default=None: getattr(cfg, attr, default)
            else:
                ctype = cfg.get('type')
                get = lambda attr, default=None: cfg.get(attr, default)

            factory = _CONSTRAINT_FACTORIES.get(ctype)
            if not factory:
                continue
            try:
                constraint = factory(get)
                self.constraints.append(constraint)
            except Exception:
                # 可添加日志记录具体错误
                continue

    def add_constraint(self, constraint: Constraint) -> None:
        """动态添加约束实例"""
        self.constraints.append(constraint)

    def project(self, df: pd.DataFrame, max_iter: int = 10) -> pd.DataFrame:
        """
        迭代调用各约束的 project 方法，直到收敛或达到最大迭代次数
        """
        df_proj = df.copy()
        for _ in range(max_iter):
            df_before = df_proj.copy()
            for constraint in self.constraints:
                df_proj = constraint.project(df_proj)
            if df_proj.equals(df_before):
                break
        return df_proj

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        评估每个约束的满足度，返回字典并包含总体平均分
        """
        scores: Dict[str, float] = {}
        for idx, constraint in enumerate(self.constraints):
            key = f"{constraint.__class__.__name__}_{idx}"
            scores[key] = constraint.evaluate(df)
        overall = sum(scores.values()) / len(scores) if scores else 1.0
        scores['overall'] = overall
        return scores

    def filter_satisfied(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回满足所有约束的行集合"""
        mask = pd.Series(True, index=df.index)
        for constraint in self.constraints:
            mask &= constraint.is_satisfied(df)
        return df[mask]