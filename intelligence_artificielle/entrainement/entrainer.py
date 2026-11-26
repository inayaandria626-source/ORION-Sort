
from pathlib import Path
import json
import tensorflow as tf

# Configuration
BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "dataset_prepare"
DOSSIER_MODELES = BASE / "modeles"

DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)

TAILLE_IMAGE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 12
SEED = 42

# Limiter l'utilisation du processeur
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

print("TensorFlow :", tf.__version__)
print("Chargement des images...")

# Chargement des trois ensembles
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "train",
    image_size=TAILLE_IMAGE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "validation",
    image_size=TAILLE_IMAGE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "test",
    image_size=TAILLE_IMAGE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

categories = train_ds.class_names
nombre_categories = len(categories)

print("Catégories :", categories)

# Sauvegarder les noms des catégories
with open(DOSSIER_MODELES / "categories.json", "w", encoding="utf-8") as fichier:
    json.dump(categories, fichier, ensure_ascii=False, indent=4)

# Préparation des images pour MobileNetV2
pretraitement = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(
    lambda images, etiquettes: (pretraitement(images), etiquettes),
    num_parallel_calls=2
).prefetch(1)

validation_ds = validation_ds.map(
    lambda images, etiquettes: (pretraitement(images), etiquettes),
    num_parallel_calls=2
).prefetch(1)

test_ds = test_ds.map(
    lambda images, etiquettes: (pretraitement(images), etiquettes),
    num_parallel_calls=2
).prefetch(1)

# Augmentation des données
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10)
])

# Base MobileNetV2 pré-entraînée sur ImageNet
print("Chargement de MobileNetV2...")

base_modele = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_modele.trainable = False

# Construction du classificateur
entree = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(entree)
x = base_modele(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)

sortie = tf.keras.layers.Dense(
    nombre_categories,
    activation="softmax"
)(x)

modele = tf.keras.Model(entree, sortie)

# Compilation
modele.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

modele.summary()

# Sauvegarder le meilleur modèle
chemin_modele = DOSSIER_MODELES / "orion_mobilenetv2.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(chemin_modele),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# Entraînement
print("\nDébut de l'entraînement...")

historique = modele.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Évaluation sur les images de test
print("\nÉvaluation finale...")

modele = tf.keras.models.load_model(chemin_modele)

perte, precision = modele.evaluate(test_ds)

print("\nRésultats du test :")
print("Perte :", round(float(perte), 4))
print("Précision globale :", round(float(precision) * 100, 2), "%")

print("\nModèle enregistré :", chemin_modele)
print("Entraînement terminé !")
