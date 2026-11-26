
from pathlib import Path
import json
import tensorflow as tf
import numpy as np

# Chemins du projet
BASE = Path(__file__).resolve().parent.parent

CHEMIN_MODELE = BASE / "modeles" / "orion_mobilenetv2.keras"
CHEMIN_CATEGORIES = BASE / "modeles" / "categories.json"

# Charger les catégories
with open(CHEMIN_CATEGORIES, "r", encoding="utf-8") as fichier:
    categories = json.load(fichier)

# Charger le modèle entraîné
print("Chargement du modèle ORION-Sort...")
modele = tf.keras.models.load_model(CHEMIN_MODELE)

# Demander une image à tester
chemin_image = input("Entre le chemin complet de ton image : ").strip().strip('"')

image_path = Path(chemin_image)

if not image_path.exists():
    print("Erreur : image introuvable.")
else:
    # Charger et préparer l'image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    tableau = tf.keras.utils.img_to_array(image)
    tableau = np.expand_dims(tableau, axis=0)

    # Prétraitement MobileNetV2
    tableau = tf.keras.applications.mobilenet_v2.preprocess_input(tableau)

    # Prédiction
    predictions = modele.predict(tableau, verbose=0)

    indice = np.argmax(predictions[0])
    materiau = categories[indice]
    confiance = float(predictions[0][indice]) * 100

    print("\n===== RÉSULTAT ORION-SORT =====")
    print("Matériau détecté :", materiau)
    print(f"Confiance : {confiance:.2f} %")
    print("================================")
