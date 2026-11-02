import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


def main():
    data = pd.read_excel("A.xlsx")
    data.columns = data.columns.map(str)
    band_columns = [
        name for name in data.columns
        if name.lower().replace("nm", "").strip().replace(".", "", 1).isdigit()
    ]
    X = data[band_columns].astype(float)
    y = data["SOC"]

    # Calculate the five band-evaluation criteria.
    icv = (X.mean() / X.std().replace(0, np.nan)).fillna(0)
    mi = pd.Series(mutual_info_regression(X, y), index=X.columns)

    spectral_entropy = {}
    for band in X.columns:
        counts, _ = np.histogram(X[band], bins="auto")
        probabilities = counts[counts > 0] / counts.sum()
        spectral_entropy[band] = -np.sum(probabilities * np.log(probabilities))

    forest = RandomForestRegressor()
    forest.fit(X, y)
    rf_importance = pd.Series(forest.feature_importances_, index=X.columns)

    correlations = X.corr().abs()
    np.fill_diagonal(correlations.values, np.nan)
    anti_redundancy = (1 - correlations.mean(axis=1)).fillna(0)

    criteria = pd.DataFrame({
        "ICV": icv,
        "MI": mi,
        "Entropy": pd.Series(spectral_entropy),
        "RF_Importance": rf_importance,
        "Anti_Redundancy": anti_redundancy,
    })

    # Normalize the criteria and determine their entropy weights.
    normalized = (
        (criteria - criteria.min())
        .div((criteria.max() - criteria.min()).replace(0, np.nan))
        .fillna(0)
    )
    probabilities = normalized.div(normalized.sum().replace(0, np.nan))
    positive_probabilities = probabilities.where(probabilities > 0)
    information_entropy = -(
        positive_probabilities * np.log(positive_probabilities)
    ).sum() / np.log(len(normalized))
    information_entropy[normalized.sum() == 0] = 1
    diversity = 1 - information_entropy
    weights = diversity / diversity.sum()

    # Compute the TOPSIS closeness and rank the candidate bands.
    column_norms = np.sqrt((normalized ** 2).sum()).replace(0, 1)
    weighted = normalized.div(column_norms).mul(weights)
    positive_ideal = weighted.max()
    negative_ideal = weighted.min()
    distance_positive = np.sqrt(((weighted - positive_ideal) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted - negative_ideal) ** 2).sum(axis=1))
    closeness = (
        distance_negative / (distance_positive + distance_negative).replace(0, np.nan)
    ).fillna(0)

    ranking = closeness.sort_values(ascending=False)
    print("Entropy weights:\n", weights)
    print("\nTOPSIS band ranking:\n", ranking)


if __name__ == "__main__":
    main()
