# from sklearn.metrics import f1_score, mean_absolute_error
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# import numpy as np
# import pandas as pd
#
# from ..data.preprocess import build_preprocessor
#
# def delta_r(
#     original: pd.DataFrame,
#     synth: pd.DataFrame,
#     target: str,
#     task: str = "classification",
# ):
#     # 1) 列划分
#     features = [c for c in original.columns if c != target]
#     num_cols = [c for c in features if original[c].dtype != "O"]
#     cat_cols = [c for c in features if original[c].dtype == "O"]
#
#     # 2) 预处理器：缺失填补 + One-Hot
#     prep = build_preprocessor(num_cols, cat_cols)
#
#     # 3) 原始数据
#     Xo = prep.fit_transform(original[features])
#     yo = original[target]
#
#     # 4) 合成数据（用同一预处理器转换）
#     Xs = prep.transform(synth[features])
#     ys = synth[target]
#
#     # 5) 训练-评估函数
#     def train_eval(X, y):
#         Xtr, Xte, ytr, yte = train_test_split(
#             X, y, test_size=0.3, random_state=42, stratify=y if task == "classification" else None
#         )
#         Model = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
#         model = Model(min_data_in_leaf=1, min_data_in_bin=1)  # 防止小样本无特征
#         model.fit(Xtr, ytr)
#         pred = model.predict(Xte)
#         if task == "classification":
#             return f1_score(yte, pred, average="macro")
#         return 1 - mean_absolute_error(yte, pred)
#
#     r0 = train_eval(Xo, yo)
#     r1 = train_eval(Xs, ys)
#     return abs(r0 - r1) / r0


from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from ..data.preprocess import build_preprocessor

# 尝试导入 CatBoost（可选）
try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed. Continuing without it.")

# 预定义的模型配置 - 专注于表格数据的最佳模型
DEFAULT_MODELS = {
    "classification": {
        # 梯度提升模型 - 表格数据的最佳选择
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        "xgboost": lambda: xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        ),
        # Histogram-based Gradient Boosting - sklearn的高效实现
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        # 随机森林及其变体
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        # 线性模型作为基线
        "logistic_regression": lambda: LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='liblinear'  # 对小数据集更稳定
        ),
    },
    "regression": {
        # 梯度提升模型
        "lightgbm": lambda: lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        "xgboost": lambda: xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        ),
        # Histogram-based Gradient Boosting
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        ),
        # 随机森林及其变体
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),
        # 线性模型作为基线
        "ridge": lambda: Ridge(
            random_state=42,
            alpha=1.0
        ),
    }
}

# 如果 CatBoost 可用，添加到模型列表
if CATBOOST_AVAILABLE:
    DEFAULT_MODELS["classification"]["catboost"] = lambda: cb.CatBoostClassifier(
        iterations=100,
        depth=6,
        random_state=42,
        verbose=False,
        allow_writing_files=False
    )
    DEFAULT_MODELS["regression"]["catboost"] = lambda: cb.CatBoostRegressor(
        iterations=100,
        depth=6,
        random_state=42,
        verbose=False,
        allow_writing_files=False
    )


def get_best_tabular_models(task: str = "classification") -> List[str]:
    """
    获取对表格数据效果最好的模型列表（按通常性能排序）

    Parameters:
    -----------
    task : str
        任务类型，'classification' 或 'regression'

    Returns:
    --------
    List[str]
        推荐的模型名称列表
    """
    if CATBOOST_AVAILABLE:
        # CatBoost 通常在有类别特征时表现最佳
        return ["catboost", "lightgbm", "xgboost", "hist_gradient_boosting", "random_forest"]
    else:
        return ["lightgbm", "xgboost", "hist_gradient_boosting", "random_forest", "extra_trees"]


def delta_r(
        original: pd.DataFrame,
        synth: pd.DataFrame,
        target: str,
        task: str = "classification",
        models: Optional[Union[List[str], Dict[str, Any]]] = None,
        return_details: bool = False,
        use_best_models: bool = True
) -> Union[float, Dict[str, Any]]:
    """
    计算基于原始数据集和生成数据集训练的模型之间的预测精度差异。
    专门优化用于表格数据的评估。

    Parameters:
    -----------
    original : pd.DataFrame
        原始数据集 T
    synth : pd.DataFrame
        生成/合成数据集 T1
    target : str
        目标变量列名
    task : str
        任务类型，'classification' 或 'regression'
    models : Optional[Union[List[str], Dict[str, Any]]]
        要使用的模型列表或模型字典。
        - 如果是列表：使用预定义模型的名称
        - 如果是字典：键为模型名称，值为模型实例或可调用对象
        - 如果为 None：使用默认的最佳表格数据模型
    return_details : bool
        如果为 True，返回详细结果（包括每个模型的 R0、R1 和 delta_r）
        如果为 False，仅返回平均 delta_r
    use_best_models : bool
        如果为 True 且 models 为 None，只使用表现最好的模型
        如果为 False 且 models 为 None，使用所有可用模型

    Returns:
    --------
    Union[float, Dict[str, Any]]
        如果 return_details=False：返回所有模型的平均 delta_r
        如果 return_details=True：返回包含详细信息的字典
    """

    # 1) 列划分
    features = [c for c in original.columns if c != target]
    num_cols = [c for c in features if original[c].dtype != "O"]
    cat_cols = [c for c in features if original[c].dtype == "O"]

    # 2) 预处理器：缺失填补 + One-Hot
    prep = build_preprocessor(num_cols, cat_cols)

    # 3) 将原始数据分割为训练集和测试集
    X_original = original[features]
    y_original = original[target]

    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X_original,
        y_original,
        test_size=0.3,
        random_state=42,
        stratify=y_original if task == "classification" else None
    )

    # 4) 预处理
    X_orig_train_prep = prep.fit_transform(X_orig_train)
    X_orig_test_prep = prep.transform(X_orig_test)

    X_synth = synth[features]
    y_synth = synth[target]
    X_synth_prep = prep.transform(X_synth)

    # 5) 确定要使用的模型
    if models is None:
        if use_best_models:
            # 只使用最佳的表格数据模型
            best_model_names = get_best_tabular_models(task)
            model_dict = {
                name: DEFAULT_MODELS[task][name]
                for name in best_model_names
                if name in DEFAULT_MODELS[task]
            }
        else:
            # 使用所有预定义模型
            model_dict = DEFAULT_MODELS[task]
    elif isinstance(models, list):
        # 使用指定的预定义模型
        available_models = DEFAULT_MODELS[task]
        model_dict = {
            name: available_models[name]
            for name in models
            if name in available_models
        }
        # 检查是否有无效的模型名称
        invalid_models = [m for m in models if m not in available_models]
        if invalid_models:
            print(
                f"Warning: Models {invalid_models} not found for {task} task. Available: {list(available_models.keys())}")
    elif isinstance(models, dict):
        # 使用用户提供的模型字典
        model_dict = models
    else:
        raise ValueError("models must be None, a list of model names, or a dictionary of models")

    # 6) 定义评估指标
    if task == "classification":
        def eval_metric(y_true, y_pred):
            return f1_score(y_true, y_pred, average="macro")
    else:  # regression
        def eval_metric(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            mean_y = np.mean(np.abs(y_true))
            if mean_y == 0:
                return 0.0
            return max(0, 1 - mae / mean_y)

    # 7) 对每个模型计算 delta_r
    results = {}

    for model_name, model_creator in model_dict.items():
        try:
            # 创建模型实例
            if callable(model_creator):
                model_m0 = model_creator()
                model_m1 = model_creator()
            else:
                # 如果直接传入模型实例，需要克隆
                from sklearn.base import clone
                model_m0 = clone(model_creator)
                model_m1 = clone(model_creator)

            # 对于 CatBoost，需要特殊处理类别特征
            if CATBOOST_AVAILABLE and isinstance(model_m0, (cb.CatBoostClassifier, cb.CatBoostRegressor)):
                # CatBoost 可以直接处理类别特征，不需要 one-hot 编码
                # 这里简化处理，仍使用预处理后的数据
                pass

            # 训练模型 M0：基于原始数据训练集
            model_m0.fit(X_orig_train_prep, y_orig_train)

            # 训练模型 M1：基于生成数据集
            model_m1.fit(X_synth_prep, y_synth)

            # 在原始数据的测试集上评估
            pred_m0 = model_m0.predict(X_orig_test_prep)
            pred_m1 = model_m1.predict(X_orig_test_prep)

            # 计算精度
            r0 = eval_metric(y_orig_test, pred_m0)
            r1 = eval_metric(y_orig_test, pred_m1)

            # 计算 ΔR = |R0 – R1| / R0
            if r0 == 0:
                delta = float('inf') if r1 != 0 else 0.0
            else:
                delta = abs(r0 - r1) / r0

            results[model_name] = {
                "R0": r0,
                "R1": r1,
                "delta_r": delta,
                "improvement": r1 - r0,  # 正值表示生成数据训练的模型更好
                "relative_improvement": (r1 - r0) / r0 if r0 != 0 else 0.0
            }

        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            results[model_name] = {
                "R0": None,
                "R1": None,
                "delta_r": None,
                "improvement": None,
                "relative_improvement": None,
                "error": str(e)
            }

    # 8) 返回结果
    if return_details:
        # 计算汇总统计
        valid_deltas = [r["delta_r"] for r in results.values()
                        if r["delta_r"] is not None and r["delta_r"] != float('inf')]

        summary = {
            "mean_delta_r": np.mean(valid_deltas) if valid_deltas else None,
            "std_delta_r": np.std(valid_deltas) if valid_deltas else None,
            "min_delta_r": np.min(valid_deltas) if valid_deltas else None,
            "max_delta_r": np.max(valid_deltas) if valid_deltas else None,
            "median_delta_r": np.median(valid_deltas) if valid_deltas else None,
            "models": results,
            "best_model": min(results.items(), key=lambda x: x[1]["delta_r"]
            if x[1]["delta_r"] is not None and x[1]["delta_r"] != float('inf')
            else float('inf'))[0] if valid_deltas else None
        }
        return summary
    else:
        # 仅返回平均 delta_r
        valid_deltas = [r["delta_r"] for r in results.values()
                        if r["delta_r"] is not None and r["delta_r"] != float('inf')]
        return np.mean(valid_deltas) if valid_deltas else float('inf')


def delta_r_multiple_runs(
        original: pd.DataFrame,
        synth: pd.DataFrame,
        target: str,
        task: str = "classification",
        models: Optional[Union[List[str], Dict[str, Any]]] = None,
        n_runs: int = 5,
        test_size: float = 0.3,
        use_best_models: bool = True
) -> Dict[str, Any]:
    """
    多次运行 delta_r 评估，使用不同的随机种子，以获得更稳定的结果。
    专门优化用于表格数据的评估。

    Parameters:
    -----------
    original : pd.DataFrame
        原始数据集
    synth : pd.DataFrame
        生成数据集
    target : str
        目标变量列名
    task : str
        任务类型
    models : Optional[Union[List[str], Dict[str, Any]]]
        要使用的模型
    n_runs : int
        运行次数
    test_size : float
        测试集比例
    use_best_models : bool
        是否只使用最佳的表格数据模型

    Returns:
    --------
    Dict[str, Any]
        包含多次运行结果的汇总统计
    """

    all_results = []

    for seed in range(n_runs):
        # 为每次运行创建新的随机种子
        np.random.seed(seed)

        # 修改原始 delta_r 函数以支持不同的随机种子
        result = _delta_r_with_seed(
            original, synth, target, task, models, seed, test_size, use_best_models
        )
        all_results.append(result)

    # 汇总结果
    model_names = list(all_results[0].keys())
    final_results = {}

    for model_name in model_names:
        model_deltas = [run[model_name]["delta_r"] for run in all_results
                        if run[model_name]["delta_r"] is not None
                        and run[model_name]["delta_r"] != float('inf')]

        model_r0s = [run[model_name]["R0"] for run in all_results
                     if run[model_name]["R0"] is not None]

        model_r1s = [run[model_name]["R1"] for run in all_results
                     if run[model_name]["R1"] is not None]

        final_results[model_name] = {
            "delta_r_mean": np.mean(model_deltas) if model_deltas else None,
            "delta_r_std": np.std(model_deltas) if model_deltas else None,
            "delta_r_min": np.min(model_deltas) if model_deltas else None,
            "delta_r_max": np.max(model_deltas) if model_deltas else None,
            "R0_mean": np.mean(model_r0s) if model_r0s else None,
            "R0_std": np.std(model_r0s) if model_r0s else None,
            "R1_mean": np.mean(model_r1s) if model_r1s else None,
            "R1_std": np.std(model_r1s) if model_r1s else None,
            "n_valid_runs": len(model_deltas)
        }

    # 找出平均 delta_r 最低的模型
    best_model = min(final_results.items(),
                     key=lambda x: x[1]["delta_r_mean"] if x[1]["delta_r_mean"] is not None else float('inf'))

    return {
        "models": final_results,
        "n_runs": n_runs,
        "test_size": test_size,
        "best_model": best_model[0],
        "best_model_delta_r": best_model[1]["delta_r_mean"]
    }


def _delta_r_with_seed(
        original: pd.DataFrame,
        synth: pd.DataFrame,
        target: str,
        task: str,
        models: Optional[Union[List[str], Dict[str, Any]]],
        seed: int,
        test_size: float,
        use_best_models: bool = True
) -> Dict[str, Any]:
    """内部函数：使用指定的随机种子运行 delta_r"""

    # 类似于 delta_r 函数，但使用指定的随机种子
    features = [c for c in original.columns if c != target]
    num_cols = [c for c in features if original[c].dtype != "O"]
    cat_cols = [c for c in features if original[c].dtype == "O"]

    from ..data.preprocess import build_preprocessor
    prep = build_preprocessor(num_cols, cat_cols)

    X_original = original[features]
    y_original = original[target]

    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X_original,
        y_original,
        test_size=test_size,
        random_state=seed,  # 使用指定的种子
        stratify=y_original if task == "classification" else None
    )

    X_orig_train_prep = prep.fit_transform(X_orig_train)
    X_orig_test_prep = prep.transform(X_orig_test)

    X_synth = synth[features]
    y_synth = synth[target]
    X_synth_prep = prep.transform(X_synth)

    # 确定模型
    if models is None:
        if use_best_models:
            best_model_names = get_best_tabular_models(task)
            model_dict = {
                name: DEFAULT_MODELS[task][name]
                for name in best_model_names
                if name in DEFAULT_MODELS[task]
            }
        else:
            model_dict = DEFAULT_MODELS[task]
    elif isinstance(models, list):
        available_models = DEFAULT_MODELS[task]
        model_dict = {
            name: available_models[name]
            for name in models
            if name in available_models
        }
    elif isinstance(models, dict):
        model_dict = models

    # 评估指标
    if task == "classification":
        def eval_metric(y_true, y_pred):
            return f1_score(y_true, y_pred, average="macro")
    else:
        def eval_metric(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            mean_y = np.mean(np.abs(y_true))
            if mean_y == 0:
                return 0.0
            return max(0, 1 - mae / mean_y)

    results = {}

    for model_name, model_creator in model_dict.items():
        try:
            if callable(model_creator):
                # 创建模型时修改随机种子
                model_m0 = _create_model_with_seed(model_creator, seed)
                model_m1 = _create_model_with_seed(model_creator, seed)
            else:
                from sklearn.base import clone
                model_m0 = clone(model_creator)
                model_m1 = clone(model_creator)

            model_m0.fit(X_orig_train_prep, y_orig_train)
            model_m1.fit(X_synth_prep, y_synth)

            pred_m0 = model_m0.predict(X_orig_test_prep)
            pred_m1 = model_m1.predict(X_orig_test_prep)

            r0 = eval_metric(y_orig_test, pred_m0)
            r1 = eval_metric(y_orig_test, pred_m1)

            if r0 == 0:
                delta = float('inf') if r1 != 0 else 0.0
            else:
                delta = abs(r0 - r1) / r0

            results[model_name] = {
                "R0": r0,
                "R1": r1,
                "delta_r": delta
            }

        except Exception as e:
            results[model_name] = {
                "R0": None,
                "R1": None,
                "delta_r": None
            }

    return results


def _create_model_with_seed(model_creator, seed):
    """创建模型并设置随机种子"""
    model = model_creator()
    # 尝试设置随机种子
    if hasattr(model, 'random_state'):
        model.random_state = seed
    elif hasattr(model, 'seed'):
        model.seed = seed
    return model


def compare_synthetic_quality(
        original: pd.DataFrame,
        synth_list: List[pd.DataFrame],
        synth_names: List[str],
        target: str,
        task: str = "classification",
        n_runs: int = 5
) -> pd.DataFrame:
    """
    比较多个合成数据集的质量

    Parameters:
    -----------
    original : pd.DataFrame
        原始数据集
    synth_list : List[pd.DataFrame]
        合成数据集列表
    synth_names : List[str]
        合成数据集的名称
    target : str
        目标变量列名
    task : str
        任务类型
    n_runs : int
        每个数据集的运行次数

    Returns:
    --------
    pd.DataFrame
        比较结果的表格
    """
    results = []

    for synth_data, name in zip(synth_list, synth_names):
        print(f"Evaluating {name}...")

        # 使用最佳模型进行评估
        run_results = delta_r_multiple_runs(
            original=original,
            synth=synth_data,
            target=target,
            task=task,
            n_runs=n_runs,
            use_best_models=True
        )

        # 提取关键指标
        best_model = run_results["best_model"]
        best_delta_r = run_results["best_model_delta_r"]

        # 计算所有模型的平均 delta_r
        all_deltas = [
            model_res["delta_r_mean"]
            for model_res in run_results["models"].values()
            if model_res["delta_r_mean"] is not None
        ]
        avg_delta_r = np.mean(all_deltas) if all_deltas else None

        results.append({
            "Dataset": name,
            "Best Model": best_model,
            "Best Delta-R": best_delta_r,
            "Avg Delta-R": avg_delta_r,
            "N Runs": n_runs
        })

    return pd.DataFrame(results).sort_values("Best Delta-R")