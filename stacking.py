
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def main():
    # 读取训练集与独立测试集
    train = pd.read_excel("train.xlsx")
    test = pd.read_excel("test.xlsx")
    X_train, y_train = train.iloc[:, 3:].to_numpy(dtype=float), train.iloc[:, 2].to_numpy(dtype=float)
    X_test, y_test = test.iloc[:, 3:].to_numpy(dtype=float), test.iloc[:, 2].to_numpy(dtype=float)

    # 第一层：基模型生成 OOF 元特征
    base_models = {"1D-CNN": None, "XGBoost": XGBRegressor,
                   "SVR": SVR, "PLSR": PLSRegression}
    meta_train = np.empty((len(X_train), len(base_models)))
    meta_test = np.empty((len(X_test), len(base_models)))
    folds = list(KFold().split(X_train))

    for j, (name, model_type) in enumerate(base_models.items()):
        # 最后一轮用完整训练集训练模型，并预测独立测试集
        for fit_idx, val_idx in folds + [(np.arange(len(X_train)), None)]:
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X_train[fit_idx])
            X_pred = scaler.transform(X_test if val_idx is None else X_train[val_idx])

            if name == "1D-CNN":
                tf.keras.backend.clear_session()
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
                    tf.keras.layers.Conv1D(16, 3, padding="same", activation="relu"),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Dense(1),
                ])
                model.compile(optimizer="adam", loss="mse")
                model.fit(X_fit[..., None], y_train[fit_idx], verbose=0)
                prediction = model.predict(X_pred[..., None], verbose=0).ravel()
            else:
                model = model_type()
                model.fit(X_fit, y_train[fit_idx])
                prediction = np.asarray(model.predict(X_pred)).ravel()

            if val_idx is None:
                meta_test[:, j] = prediction
            else:
                meta_train[val_idx, j] = prediction

    # 第二层：RF 元学习器融合基模型预测
    meta_model = RandomForestRegressor()
    meta_model.fit(meta_train, y_train)
    y_pred = meta_model.predict(meta_test)

    # 独立测试集评价
    print("R²:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("Bias:", np.mean(y_pred - y_test))


if __name__ == "__main__":
    main()
