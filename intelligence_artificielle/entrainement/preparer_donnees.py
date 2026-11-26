
from pathlib import Path
import random
import shutil

# Dossier contenant les images originales
BASE = Path(__file__).resolve().parent.parent / "dataset"

# Dossier qui recevra les images réparties
DESTINATION = Path(__file__).resolve().parent.parent / "dataset_prepare"

CATEGORIES = ["carton", "metal", "papier", "plastique", "vide"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(42)

# Ne pas écraser une préparation existante
if DESTINATION.exists():
    raise FileExistsError(
        "Le dossier dataset_prepare existe déjà. "
        "Ne le supprime pas avant de vérifier son contenu."
    )

for ensemble in ["train", "validation", "test"]:
    for categorie in CATEGORIES:
        (DESTINATION / ensemble / categorie).mkdir(
            parents=True,
            exist_ok=True
        )

total_general = 0

for categorie in CATEGORIES:

    dossier = BASE / categorie

    images = [
        fichier for fichier in dossier.rglob("*")
        if fichier.is_file()
        and fichier.suffix.lower() in EXTENSIONS
    ]

    random.shuffle(images)

    nombre = len(images)
    nombre_test = round(nombre * 0.15)
    nombre_validation = round(nombre * 0.15)

    images_test = images[:nombre_test]

    images_validation = images[
        nombre_test:nombre_test + nombre_validation
    ]

    images_train = images[
        nombre_test + nombre_validation:
    ]

    repartition = {
        "train": images_train,
        "validation": images_validation,
        "test": images_test
    }

    for ensemble, liste_images in repartition.items():

        destination = DESTINATION / ensemble / categorie

        for image in liste_images:
            shutil.copy2(
                image,
                destination / image.name
            )

    print(
        categorie,
        "| Total :", nombre,
        "| Entraînement :", len(images_train),
        "| Validation :", len(images_validation),
        "| Test :", len(images_test)
    )

    total_general += nombre

print("\nPréparation terminée.")
print("Total des images :", total_general)
print("Dossier créé :", DESTINATION)
