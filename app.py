import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from retrieval import search_images


st.set_page_config(
    page_title="Système CBIR - GLCM",
    page_icon="🖼️",
    layout="wide"
)


# ==================================================
# CONFIGURATION
# ==================================================

LEVELS = 8
DISTANCE = 1
ANGLES = [0, 45, 90, 135]

DESCRIPTORS = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "mean",
    "uniformity",
    "entropy"
]

INDEX_FILE = ROOT / "features" / "glcm_features.csv"


# ==================================================
# STYLE
# ==================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

h1 {
    font-size: 1.65rem !important;
}

h2 {
    font-size: 1.3rem !important;
}

h3 {
    font-size: 1rem !important;
}

.st-key-offline_panel {
    background-color: #EAF3FC;
    border: 1px solid #C8DCEC;
    border-radius: 12px;
    padding: 16px;
}

.st-key-query_panel {
    background-color: #EAF5ED;
    border: 1px solid #C9E0CF;
    border-radius: 12px;
    padding: 16px;
}

.st-key-results_panel {
    background-color: #EAF5ED;
    border: 1px solid #C9E0CF;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.75);
    padding: 8px;
    border-radius: 8px;
}

div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.72);
    border-radius: 8px;
    border: 1px solid rgba(100,100,100,0.12);
}

.phase-offline-title {
    color: #234C70;
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 12px;
}

.phase-online-title {
    color: #285C3B;
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 12px;
}

.panel-subtitle {
    color: #333333;
    font-weight: 700;
    font-size: 1.05rem;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# TITRE
# ==================================================

st.title(
    "SYSTÈME D'INDEXATION ET DE RECHERCHE DES IMAGES PAR GLCM"
)

st.caption(
    "Content-Based Image Retrieval | Analyse des textures et recherche par similarité"
)

st.divider()


# ==================================================
# ORGANISATION
# ==================================================

offline_col, online_col = st.columns(
    [0.85, 3.25],
    gap="medium"
)


# ==================================================
# PHASE HORS-LIGNE
# ==================================================

with offline_col:

    with st.container(height=760, key="offline_panel"):

        st.markdown(
            '<div class="phase-offline-title">PHASE HORS-LIGNE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="panel-subtitle">01. Paramètres et indexation</div>',
            unsafe_allow_html=True
        )

        st.divider()

        with st.expander("Quantification", expanded=True):

            st.metric("Niveaux de gris", LEVELS)

            st.caption("Image originale : 256 niveaux")
            st.caption("Image quantifiée : 8 niveaux")

        with st.expander("Configuration GLCM", expanded=True):

            st.metric("Distance", f"{DISTANCE} pixel")
            st.metric("Nombre d'orientations", len(ANGLES))

            st.markdown("**Orientations utilisées**")

            for angle in ANGLES:
                st.write(f"• {angle}°")

        with st.expander("Dimensions des caractéristiques"):

            st.write("**Matrice GLCM :** 8 × 8")
            st.write("**Caractéristiques par orientation :** 8")
            st.write("**Nombre d'orientations :** 4")

            st.metric("Taille du vecteur", "32 caractéristiques")

        with st.expander("Descripteurs GLCM"):

            for descriptor in DESCRIPTORS:
                st.write(f"• {descriptor.capitalize()}")

        with st.expander("Base de données DTD", expanded=True):

            if INDEX_FILE.exists():

                index_df = pd.read_csv(INDEX_FILE)

                st.metric("Images indexées", len(index_df))

                if "label" in index_df.columns:
                    st.metric(
                        "Catégories",
                        index_df["label"].nunique()
                    )

                st.success("Index disponible")

            else:
                st.warning("Fichier d'indexation introuvable")

        st.divider()

        st.caption(
            "Extraction et stockage des caractéristiques de la base DTD."
        )


# ==================================================
# PHASE EN LIGNE
# ==================================================

with online_col:

    st.markdown(
        '<div class="phase-online-title">PHASE EN LIGNE</div>',
        unsafe_allow_html=True
    )

    center, right = st.columns(
        [2.0, 1.25],
        gap="medium"
    )


    # ==================================================
    # IMAGE REQUETE ET MATRICES
    # ==================================================

    with center:

        with st.container(height=760, key="query_panel"):

            st.markdown(
                '<div class="panel-subtitle">02. Image requête et analyse GLCM</div>',
                unsafe_allow_html=True
            )

            st.divider()

            uploaded = st.file_uploader(
                "Importer une image",
                type=["jpg", "jpeg", "png", "bmp"]
            )

            if uploaded:

                image_bytes = uploaded.getvalue()

                st.image(
                    image_bytes,
                    caption=uploaded.name,
                    width=260
                )

                top_k = st.selectbox(
                    "Nombre d'images similaires",
                    [5, 10, 15, 20, 30, 50],
                    index=0
                )

                search_button = st.button(
                    "Rechercher les images similaires",
                    type="primary",
                    use_container_width=True
                )

                if search_button:

                    suffix = Path(uploaded.name).suffix or ".jpg"

                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False
                    ) as temp:

                        temp.write(image_bytes)
                        query_path = Path(temp.name)

                    try:

                        with st.spinner(
                            "Analyse GLCM et recherche en cours..."
                        ):

                            results, details = search_images(
                                query_path,
                                top_k
                            )

                        st.session_state["query_details"] = details
                        st.session_state["results"] = results
                        st.session_state["query_name"] = uploaded.name

                        st.success("Recherche terminée")

                    except Exception as error:

                        st.error(f"Erreur : {error}")

                    finally:

                        query_path.unlink(missing_ok=True)

            else:

                st.info(
                    "Importez une image pour commencer l'analyse."
                )


            # MATRICES GLCM BRUTES

            if "query_details" in st.session_state:

                details = st.session_state["query_details"]

                st.divider()

                st.markdown("**Analyse des matrices GLCM**")

                st.caption(
                    "Matrices brutes : nombres de cooccurrences sans normalisation."
                )

                def color_nonzero(value):

                    if value != 0:
                        return (
                            "background-color: #F8B4B4;"
                            "color: #990000;"
                            "font-weight: bold;"
                        )

                    return (
                        "background-color: white;"
                        "color: black;"
                    )

                for angle in ANGLES:

                    with st.expander(
                        f"Distance {DISTANCE} — Orientation {angle}°",
                        expanded=(angle == 0)
                    ):

                        matrix = details["matrices"][angle]

                        matrix_df = pd.DataFrame(
                            matrix,
                            index=range(LEVELS),
                            columns=range(LEVELS)
                        )

                        st.caption(
                            "Matrice de cooccurrence brute (sans normalisation)"
                        )

                        st.dataframe(
                            matrix_df.style
                            .format("{:.0f}")
                            .map(color_nonzero),
                            use_container_width=True,
                            height=240
                        )

                        st.caption(
                            "Descripteurs calculés sur la matrice normalisée"
                        )

                        values = {
                            name: details["features"][f"{name}_{angle}"]
                            for name in DESCRIPTORS
                        }

                        feature_df = pd.DataFrame(
                            list(values.items()),
                            columns=["Descripteur", "Valeur"]
                        )

                        st.dataframe(
                            feature_df.style.format(
                                {"Valeur": "{:.5f}"}
                            ),
                            hide_index=True,
                            use_container_width=True,
                            height=300
                        )


    # ==================================================
    # RESULTATS
    # ==================================================

    with right:

        with st.container(height=760, key="results_panel"):

            st.markdown(
                '<div class="panel-subtitle">03. Images similaires retrouvées</div>',
                unsafe_allow_html=True
            )

            st.divider()

            if "results" in st.session_state:

                results = st.session_state["results"]

                st.caption(
                    f"{len(results)} images similaires retrouvées"
                )

                for rank, (_, row) in enumerate(
                    results.iterrows(),
                    start=1
                ):

                    image_path = ROOT / row["image_path"]

                    with st.container(border=True):

                        st.markdown(
                            f"**#{rank} — {row['label']}**"
                        )

                        if image_path.exists():

                            st.image(
                                str(image_path),
                                use_container_width=True
                            )

                        st.metric(
                            "Similarité indicative",
                            f"{row['similarity']:.2f}%"
                        )

                        st.caption(
                            f"Distance euclidienne : {row['distance']:.5f}"
                        )

                        with st.expander(
                            "Détails des descripteurs"
                        ):

                            st.caption(
                                f"Fichier : {image_path.name}"
                            )

                            for angle in ANGLES:

                                st.markdown(
                                    f"**Orientation {angle}°**"
                                )

                                values = [
                                    row[f"{name}_{angle}"]
                                    for name in DESCRIPTORS
                                ]

                                feature_df = pd.DataFrame({
                                    "Descripteur": DESCRIPTORS,
                                    "Valeur": values
                                })

                                st.dataframe(
                                    feature_df.style.format(
                                        {"Valeur": "{:.5f}"}
                                    ),
                                    hide_index=True,
                                    use_container_width=True
                                )

            else:

                st.info(
                    "Les images retrouvées apparaîtront ici après la recherche."
                )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "CBIR — Content-Based Image Retrieval | "
    "GLCM Texture Descriptors | Academic Project"
)
