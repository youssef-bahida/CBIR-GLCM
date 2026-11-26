import numpy as np
from skimage.feature import graycomatrix, graycoprops

from preprocessing import load_image, quantize_image

LEVELS = 8
DISTANCE = 1

ANGLES = [0, 45, 90, 135]

PROPERTIES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation"
]


def extract_glcm_details(image_path):

    image = load_image(image_path)
    quantized = quantize_image(image, LEVELS)

    glcm_raw = graycomatrix(
        quantized,
        distances=[DISTANCE],
        angles=[np.deg2rad(a) for a in ANGLES],
        levels=LEVELS,
        symmetric=True,
        normed=False
    )

    glcm_normalized = glcm_raw.astype(np.float64)

    for i in range(len(ANGLES)):
        total = glcm_normalized[:, :, 0, i].sum()

        if total > 0:
            glcm_normalized[:, :, 0, i] /= total

    matrices = {}
    features = {}

    levels = np.arange(LEVELS)

    for i, angle in enumerate(ANGLES):

        matrices[angle] = glcm_raw[:, :, 0, i]

        normalized_matrix = glcm_normalized[:, :, 0, i]

        values = {}

        for prop in PROPERTIES:
            values[prop] = float(
                np.nan_to_num(
                    graycoprops(glcm_normalized, prop)[0, i]
                )
            )

        px = normalized_matrix.sum(axis=1)
        py = normalized_matrix.sum(axis=0)

        mean = (
            np.sum(levels * px) +
            np.sum(levels * py)
        ) / 2

        uniformity = np.sum(normalized_matrix ** 2)

        nonzero = normalized_matrix[normalized_matrix > 0]

        entropy = -np.sum(nonzero * np.log2(nonzero))

        values["mean"] = float(mean)
        values["uniformity"] = float(uniformity)
        values["entropy"] = float(entropy)

        for name, value in values.items():
            features[f"{name}_{angle}"] = value

    return {
        "original": image,
        "quantized": quantized,
        "matrices": matrices,
        "features": features
    }


def extract_features(image_path):
    return extract_glcm_details(image_path)["features"]
