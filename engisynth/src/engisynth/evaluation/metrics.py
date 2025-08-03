from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
import pandas as pd

from engisynth.data.preprocess import build_preprocessor

def delta_r(
    original: pd.DataFrame,
    synth: pd.DataFrame,
    target: str,
    task: str = "classification",
):
    # 1) 列划分
    features = [c for c in original.columns if c != target]
    num_cols = [c for c in features if original[c].dtype != "O"]
    cat_cols = [c for c in features if original[c].dtype == "O"]

    # 2) 预处理器：缺失填补 + One-Hot
    prep = build_preprocessor(num_cols, cat_cols)

    # 3) 原始数据
    Xo = prep.fit_transform(original[features])
    yo = original[target]

    # 4) 合成数据（用同一预处理器转换）
    Xs = prep.transform(synth[features])
    ys = synth[target]

    # 5) 训练-评估函数
    def train_eval(X, y):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if task == "classification" else None
        )
        Model = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        model = Model(min_data_in_leaf=1, min_data_in_bin=1)  # 防止小样本无特征
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        if task == "classification":
            return f1_score(yte, pred, average="macro")
        return 1 - mean_absolute_error(yte, pred)

    r0 = train_eval(Xo, yo)
    r1 = train_eval(Xs, ys)
    return abs(r0 - r1) / r0

