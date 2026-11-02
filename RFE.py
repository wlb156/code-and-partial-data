
import pandas as pd
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor


def main():
    data = pd.read_excel("training set.xlsx")
    X = data.drop(columns=["序号", "SOC"])
    y = data["SOC"]

    model = XGBRegressor()
    rfe = RFE(estimator=model)
    rfe.fit(X, y)

    selected_bands = X.columns[rfe.support_]
    print("Selected bands:", selected_bands.tolist())


if __name__ == "__main__":
    main()
