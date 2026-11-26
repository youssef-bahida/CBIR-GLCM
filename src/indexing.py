import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glcm_extraction import extract_features

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "dataset" / "dtd" / "images"
OUTPUT_FILE = ROOT / "features" / "glcm_features.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_index():

    images = [
        p for p in IMAGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    records = []

    print(f"Images trouvees : {len(images)}")

    for image_path in tqdm(images, desc="GLCM Indexation"):

        try:
            features = extract_features(image_path)

            record = {
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "label": image_path.parent.name
            }

            record.update(features)
            records.append(record)

        except Exception as error:
            print(f"Erreur : {image_path} : {error}")

    if records:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(records).to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(f"\nIndexation terminee : {len(records)} images")
        print(f"Caracteristiques : {len(features)}")
        print(f"Fichier : {OUTPUT_FILE}")


if __name__ == "__main__":
    build_index()
