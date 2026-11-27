
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parent.parent

CHEMIN_MODELE = BASE / "modeles" / "orion_mobilenetv2_v2.keras"
CHEMIN_TEST = BASE / "dataset_prepare" / "test"
DOSSIER_RESULTATS = BASE / "entrainement" / "resultats_evaluation"

DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)

print("Chargement du modèle ORION-Sort V2...")
modele = tf.keras.models.load_model(CHEMIN_MODELE)

print("Chargement des images de test...")

dataset = tf.keras.utils.image_dataset_from_directory(
    CHEMIN_TEST,
    image_size=(224, 224),
    batch_size=8,
    shuffle=False
)

categories = dataset.class_names

images_reelles = []
predictions = []

for images, labels in dataset:
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)

    resultats = modele.predict(images, verbose=0)
    predictions.extend(np.argmax(resultats, axis=1))
    images_reelles.extend(labels.numpy())

print("\n========== ÉVALUATION ORION-SORT V2 ==========")
print("Nombre d'images testées :", len(images_reelles))

precision = np.mean(np.array(images_reelles) == np.array(predictions))
print(f"Précision globale : {precision * 100:.2f} %")

print("\nRapport de classification :")
print(classification_report(
    images_reelles,
    predictions,
    target_names=categories,
    digits=2
))

matrice = confusion_matrix(images_reelles, predictions)

affichage = ConfusionMatrixDisplay(
    confusion_matrix=matrice,
    display_labels=categories
)

affichage.plot(cmap="Blues", values_format="d")
plt.title("Matrice de confusion - ORION-Sort V2")
plt.xticks(rotation=45)
plt.tight_layout()

chemin_graphique = DOSSIER_RESULTATS / "matrice_confusion_v2.png"
plt.savefig(chemin_graphique)

plt.close()

print("\nMatrice V2 enregistrée :", chemin_graphique)
