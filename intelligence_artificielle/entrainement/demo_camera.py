
import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Limiter l'utilisation du processeur
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Chemins du projet
BASE = Path(__file__).resolve().parents[1]

CHEMIN_MODELE = BASE / "modeles" / "orion_mobilenetv2_v2.keras"
CHEMIN_TRAIN = BASE / "dataset_prepare" / "train"

# Récupérer les catégories dans le même ordre que l'entraînement
categories = sorted([
    dossier.name
    for dossier in CHEMIN_TRAIN.iterdir()
    if dossier.is_dir()
])

print("Chargement du modèle ORION-Sort V2...")
modele = tf.keras.models.load_model(CHEMIN_MODELE)
print("Modèle chargé !")
print("Catégories :", categories)

# Ouvrir la caméra
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERREUR : impossible d'ouvrir la caméra.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prediction_actuelle = "En attente..."
confiance_actuelle = 0
compteur = 0

print("Caméra activée. Appuie sur Q pour quitter.")

while True:
    succes, image = camera.read()

    if not succes:
        print("Impossible de lire l'image de la caméra.")
        break

    compteur = compteur + 1

    # Faire une prédiction toutes les 5 images
    if compteur % 5 == 0:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_redimensionnee = cv2.resize(image_rgb, (224, 224))

        tableau = np.array(image_redimensionnee, dtype=np.float32)
        tableau = preprocess_input(tableau)
        tableau = np.expand_dims(tableau, axis=0)

        resultats = modele.predict(tableau, verbose=0)

        indice = np.argmax(resultats[0])
        prediction_actuelle = categories[indice]
        confiance_actuelle = float(resultats[0][indice]) * 100

    # Afficher le résultat
    cv2.rectangle(image, (10, 10), (430, 100), (20, 20, 20), -1)

    texte = "Matiere : " + prediction_actuelle
    cv2.putText(
        image, texte, (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0), 2
    )

    texte_confiance = "Confiance : " + str(round(confiance_actuelle, 2)) + "%"
    cv2.putText(
        image, texte_confiance, (25, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2
    )

    cv2.putText(
        image, "Appuie sur Q pour quitter",
        (15, 460), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )

    cv2.imshow("ORION-Sort - Demonstration IA", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Demonstration terminee.")
