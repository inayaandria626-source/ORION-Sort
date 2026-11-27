
import cv2
from pathlib import Path

# Dossier où enregistrer les nouvelles photos
BASE = Path(__file__).resolve().parents[1]
DOSSIER = BASE / "dataset" / "plastique"

DOSSIER.mkdir(parents=True, exist_ok=True)

# Compter les photos déjà présentes
nombre_initial = len(list(DOSSIER.glob("*.jpg")))
compteur = nombre_initial

# Objectif de nouvelles photos
objectif = 150
photos_ajoutees = 0

# Ouvrir la caméra
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Impossible d'ouvrir la caméra.")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("CAMERA ORION-SORT ACTIVEE")
print("S = enregistrer une photo")
print("Q = quitter")

while True:
    succes, image = camera.read()

    if not succes:
        print("Erreur de lecture de la caméra.")
        break

    # Afficher les instructions
    cv2.putText(
        image,
        "S : ENREGISTRER | Q : QUITTER",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    texte = "Photos ajoutees : " + str(photos_ajoutees) + "/" + str(objectif)

    cv2.putText(
        image,
        texte,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("ORION-Sort - Capture plastique", image)

    touche = cv2.waitKey(1) & 0xFF

    if touche == ord("s"):
        if photos_ajoutees < objectif:
            compteur = compteur + 1
            nom = "plastique_camera_" + str(compteur) + ".jpg"
            chemin = DOSSIER / nom

            cv2.imwrite(str(chemin), image)
            photos_ajoutees = photos_ajoutees + 1

            print("Photo enregistree :", nom)

        else:
            print("Objectif de 150 photos atteint !")

    elif touche == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Capture terminee.")
print("Nouvelles photos :", photos_ajoutees)
print("Dossier :", DOSSIER)
