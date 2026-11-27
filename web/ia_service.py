
import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Limiter l'utilisation du processeur
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# Chemins du projet
BASE = Path(__file__).resolve().parents[1]

CHEMIN_MODELE = (
    BASE / "intelligence_artificielle"
    / "modeles"
    / "orion_mobilenetv2_v2.keras"
)

CHEMIN_TRAIN = (
    BASE / "intelligence_artificielle"
    / "dataset_prepare"
    / "train"
)


# Variables du modèle
modele = None
categories = None


# Charger le modèle une seule fois
def charger_modele():

    global modele
    global categories

    if modele is None:

        if not CHEMIN_MODELE.exists():
            raise FileNotFoundError("Modèle V2 introuvable.")

        if not CHEMIN_TRAIN.exists():
            raise FileNotFoundError("Dossier dataset_prepare/train introuvable.")

        categories = sorted([
            dossier.name
            for dossier in CHEMIN_TRAIN.iterdir()
            if dossier.is_dir()
        ])

        print("Chargement du modèle ORION-Sort V2...")

        modele = tf.keras.models.load_model(
            CHEMIN_MODELE
        )

        print("Modèle chargé !")
        print("Catégories :", categories)

    return modele


# Analyser une image
def predire_image(image_bytes):

    modele_charge = charger_modele()

    # Transformer les données reçues en image
    tableau_bytes = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        tableau_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Image invalide ou illisible.")

    # Même traitement que demo_camera.py
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_redimensionnee = cv2.resize(
        image_rgb,
        (224, 224)
    )

    tableau = np.array(
        image_redimensionnee,
        dtype=np.float32
    )

    tableau = preprocess_input(tableau)

    tableau = np.expand_dims(
        tableau,
        axis=0
    )

    # Prédiction
    resultats = modele_charge.predict(
        tableau,
        verbose=0
    )

    indice = int(np.argmax(resultats[0]))

    materiau = categories[indice]

    confiance = float(resultats[0][indice]) * 100

    return {
        "materiau": materiau,
        "confiance": round(confiance, 2),
        "version_modele": "ORION-Sort V2"
    }
