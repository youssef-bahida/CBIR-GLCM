import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glcm_extraction import extract_glcm_details

ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "features" / "glcm_features.csv"


def search_images(query_path, top_k=5):

    df = pd.read_csv(INDEX_FILE)

    columns = [
        c for c in df.columns
        if c not in ["image_path", "label"]
    ]

    database = df[columns].values

    query_details = extract_glcm_details(query_path)

    query = np.array([
        query_details["features"][c]
        for c in columns
    ]).reshape(1, -1)

    scaler = StandardScaler()

    database_scaled = scaler.fit_transform(database)
    query_scaled = scaler.transform(query)

    distances = euclidean_distances(
        query_scaled,
        database_scaled
    )[0]

    results = df.copy()
    results["distance"] = distances

    results["similarity"] = (
        100 * np.exp(
            -distances / np.sqrt(len(columns))
        )
    )

    query_path = Path(query_path).resolve()

    results = results[
        results["image_path"].apply(
            lambda p: (ROOT / p).resolve() != query_path
        )
    ]

    results = results.sort_values(
        "distance"
    ).head(top_k)

    return results.reset_index(drop=True), query_details
