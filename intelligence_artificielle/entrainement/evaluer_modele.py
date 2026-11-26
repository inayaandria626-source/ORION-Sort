
from pathlib import Path
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# 1. Définir les chemins
BASE = Path(__file__).resolve().parent.parent

CHEMIN_MODELE = BASE / "modeles" / "orion_mobilenetv2.keras"
CHEMIN_CATEGORIES = BASE / "modeles" / "categories.json"
DOSSIER_TEST = BASE / "dataset_prepare" / "test"

# 2. Charger les catégories
with open(CHEMIN_CATEGORIES, "r", encoding="utf-8") as fichier:
    categories = json.load(fichier)

# 3. Charger le modèle
print("Chargement du modèle ORION-Sort...")
modele = tf.keras.models.load_model(CHEMIN_MODELE)

# 4. Charger les images de test
print("Chargement des images de test...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    DOSSIER_TEST,
    image_size=(224, 224),
    batch_size=8,
    shuffle=False,
    label_mode="int"
)

# 5. Préparer les données
vraies_classes = []
predictions_classes = []

for images, etiquettes in test_ds:
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)

    predictions = modele.predict(images, verbose=0)

    predictions_classes.extend(np.argmax(predictions, axis=1))
    vraies_classes.extend(etiquettes.numpy())

vraies_classes = np.array(vraies_classes)
predictions_classes = np.array(predictions_classes)

# 6. Calculer la précision globale
precision = accuracy_score(vraies_classes, predictions_classes)

print("\n========== ÉVALUATION ORION-SORT ==========")
print(f"Nombre d'images testées : {len(vraies_classes)}")
print(f"Précision globale : {precision * 100:.2f} %")

# 7. Rapport détaillé par catégorie
print("\nRapport de classification :")

print(classification_report(
    vraies_classes,
    predictions_classes,
    labels=list(range(len(categories))),
    target_names=categories,
    zero_division=0
))

# 8. Créer la matrice de confusion
matrice = confusion_matrix(
    vraies_classes,
    predictions_classes,
    labels=list(range(len(categories)))
)

affichage = ConfusionMatrixDisplay(
    confusion_matrix=matrice,
    display_labels=categories
)

affichage.plot(
    cmap="Blues",
    xticks_rotation=45,
    values_format="d"
)

plt.title("Matrice de confusion - ORION-Sort")
plt.tight_layout()

# 9. Enregistrer le graphique
DOSSIER_RESULTATS = BASE / "resultats_evaluation"
DOSSIER_RESULTATS.mkdir(exist_ok=True)

chemin_graphique = DOSSIER_RESULTATS / "matrice_confusion.png"

plt.savefig(chemin_graphique, dpi=150)
print("\nMatrice enregistrée :", chemin_graphique)

plt.show()

print("\nÉvaluation terminée !")
