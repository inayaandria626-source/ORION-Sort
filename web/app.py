
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import unicodedata

from database import (
    obtenir_materiaux,
    obtenir_statistiques,
    enregistrer_classification,
    obtenir_classifications
)

from ia_service import predire_image
from detection_service import detecter_objet


app = Flask(__name__)

app.secret_key = "orion_sort_local"

SEUIL_CONFIANCE = 70


# Normaliser les noms des materiaux
def normaliser_nom(nom):

    nom = unicodedata.normalize("NFD", nom.strip().lower())

    return "".join(
        caractere
        for caractere in nom
        if unicodedata.category(caractere) != "Mn"
    )


# Retrouver l'identifiant du materiau dans MySQL
def trouver_materiau_id(nom_materiau):

    materiaux = obtenir_materiaux()

    nom_recherche = normaliser_nom(nom_materiau)

    for materiau in materiaux:

        nom_base = normaliser_nom(materiau["nom"])

        if nom_base == nom_recherche:
            return materiau["id"]

    return None


# Dashboard principal
@app.route("/")
def accueil():

    statistiques = obtenir_statistiques()
    classifications = obtenir_classifications()
    materiaux = obtenir_materiaux()

    return render_template(
        "index.html",
        statistiques=statistiques,
        classifications=classifications,
        materiaux=materiaux
    )


# Ajouter une classification manuellement
@app.route("/ajouter-classification", methods=["POST"])
def ajouter_classification():

    try:

        nom_materiau = request.form.get("materiau")
        confiance = float(request.form.get("confiance", 100))

        materiau_id = trouver_materiau_id(nom_materiau)

        if materiau_id is None:
            raise ValueError("Materiau introuvable : " + str(nom_materiau))

        enregistrer_classification(
            materiau_id,
            confiance,
            None,
            "Manuel"
        )

        flash("Classification manuelle enregistree.")

    except Exception as erreur:

        flash("Erreur : " + str(erreur))

    return redirect(url_for("accueil"))


# Classification d'une image envoyee par formulaire
@app.route("/classifier-image", methods=["POST"])
def classifier_image():

    try:

        fichier = request.files.get("image")

        if fichier is None:
            flash("Aucune image selectionnee.")
            return redirect(url_for("accueil"))

        image_bytes = fichier.read()

        if not image_bytes:
            flash("Image vide.")
            return redirect(url_for("accueil"))

        resultat = predire_image(image_bytes)

        materiau = resultat["materiau"]
        confiance = resultat["confiance"]
        version = resultat["version_modele"]

        if confiance < SEUIL_CONFIANCE:

            flash(
                "Resultat incertain : "
                + materiau
                + " ("
                + str(confiance)
                + " %). Aucun tri automatique autorise."
            )

            return redirect(url_for("accueil"))

        materiau_id = trouver_materiau_id(materiau)

        if materiau_id is None:
            raise ValueError("Materiau introuvable : " + materiau)

        enregistrer_classification(
            materiau_id,
            confiance,
            None,
            version
        )

        flash(
            "Classification reussie : "
            + materiau
            + " avec "
            + str(confiance)
            + " % de confiance."
        )

    except Exception as erreur:

        flash("Erreur de classification : " + str(erreur))

    return redirect(url_for("accueil"))


# Classification automatique depuis la webcam
@app.route("/api/classifier-camera", methods=["POST"])
def classifier_camera():

    try:

        fichier = request.files.get("image")

        if fichier is None:
            return jsonify({
                "succes": False,
                "message": "Aucune image recue."
            }), 400

        image_bytes = fichier.read()

        if not image_bytes:
            return jsonify({
                "succes": False,
                "message": "L'image recue est vide."
            }), 400

        resultat = predire_image(image_bytes)

        materiau = resultat["materiau"]
        confiance = resultat["confiance"]
        version = resultat["version_modele"]

        if confiance < SEUIL_CONFIANCE:

            return jsonify({
                "succes": True,
                "enregistre": False,
                "materiau": materiau,
                "confiance": confiance,
                "message": "Resultat incertain. Presentez a nouveau l'objet."
            })

        materiau_id = trouver_materiau_id(materiau)

        if materiau_id is None:
            raise ValueError("Materiau introuvable : " + materiau)

        enregistrer_classification(
            materiau_id,
            confiance,
            None,
            version
        )

        return jsonify({
            "succes": True,
            "enregistre": True,
            "materiau": materiau,
            "confiance": confiance,
            "message": "Classification reussie."
        })

    except Exception as erreur:

        return jsonify({
            "succes": False,
            "message": str(erreur)
        }), 500


# Detection automatique des objets avec OpenCV
@app.route("/api/detecter-camera", methods=["POST"])
def detecter_camera():

    try:

        fichier = request.files.get("image")

        if fichier is None:
            return jsonify({
                "succes": False,
                "message": "Aucune image recue."
            }), 400

        image_bytes = fichier.read()

        if not image_bytes:
            return jsonify({
                "succes": False,
                "message": "Image vide."
            }), 400

        resultat = detecter_objet(image_bytes)

        return jsonify({
            "succes": True,
            "statut": resultat["statut"],
            "message": resultat["message"],
            "boite": resultat.get("boite"),
            "largeur": resultat["largeur"],
            "hauteur": resultat["hauteur"]
        })

    except Exception as erreur:

        return jsonify({
            "succes": False,
            "message": str(erreur)
        }), 500


if __name__ == "__main__":

    app.run(debug=True)
