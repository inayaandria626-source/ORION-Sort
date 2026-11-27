
from pathlib import Path
import json
import tensorflow as tf

# 1. Chemins du projet
BASE = Path(__file__).resolve().parent.parent

CHEMIN_MODELE = BASE / "modeles" / "orion_mobilenetv2_v2.keras"
DOSSIER_MODELES = BASE / "modeles"
DOSSIER_DATASET = BASE / "dataset_prepare"

# 2. Paramètres
TAILLE_IMAGE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 8

tf.keras.utils.set_random_seed(42)

# Limiter l'utilisation du processeur
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# 3. Charger les catégories
with open(DOSSIER_MODELES / "categories.json", "r", encoding="utf-8") as fichier:
    categories = json.load(fichier)

print("Catégories :", categories)

# 4. Charger les données
print("\nChargement des données...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DOSSIER_DATASET / "train",
    image_size=TAILLE_IMAGE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DOSSIER_DATASET / "validation",
    image_size=TAILLE_IMAGE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

if train_ds.class_names != categories:
    raise ValueError("L'ordre des catégories ne correspond pas au modèle.")

# 5. Prétraitement MobileNetV2
pretraitement = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(
    lambda images, etiquettes: (pretraitement(images), etiquettes),
    num_parallel_calls=1
).prefetch(1)

validation_ds = validation_ds.map(
    lambda images, etiquettes: (pretraitement(images), etiquettes),
    num_parallel_calls=1
).prefetch(1)

# 6. Charger la première version
print("\nChargement du modèle V1...")
modele = tf.keras.models.load_model(CHEMIN_MODELE)

# 7. Trouver MobileNetV2 dans le modèle
base_model = None

for couche in modele.layers:
    if isinstance(couche, tf.keras.Model) and "mobilenetv2" in couche.name.lower():
        base_model = couche
        break

if base_model is None:
    raise ValueError("Impossible de trouver la couche MobileNetV2.")

print("Base trouvée :", base_model.name)

# 8. Débloquer uniquement les 30 dernières couches
base_model.trainable = True

for couche in base_model.layers:
    couche.trainable = False

for couche in base_model.layers[-30:]:
    if not isinstance(couche, tf.keras.layers.BatchNormalization):
        couche.trainable = True

nombre_couches = sum(1 for couche in base_model.layers if couche.trainable)

print("Nombre de couches MobileNetV2 entraînables :", nombre_couches)

# 9. Recompiler avec un taux d'apprentissage faible
modele.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 10. Sauvegarder la meilleure version V2
chemin_v2 = DOSSIER_MODELES / "orion_mobilenetv2_v2.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(chemin_v2),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# 11. Lancer le Fine-tuning
print("\nDébut du Fine-tuning ORION-Sort...")

historique = modele.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nFine-tuning terminé !")
print("Nouvelle version :", chemin_v2)
