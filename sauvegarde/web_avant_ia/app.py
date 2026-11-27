
from flask import Flask, render_template, request, redirect, url_for, flash

from database import (
    obtenir_materiaux,
    obtenir_statistiques,
    enregistrer_classification,
    obtenir_classifications
)

app = Flask(__name__)

app.secret_key = "orion_sort_local"


@app.route("/")
def accueil():

    materiaux = obtenir_materiaux()
    statistiques = obtenir_statistiques()
    classifications = obtenir_classifications()

    return render_template(
        "index.html",
        materiaux=materiaux,
        statistiques=statistiques,
        classifications=classifications
    )


@app.route("/ajouter-classification", methods=["POST"])
def ajouter_classification():

    try:
        materiau_id = int(request.form["materiau_id"])
        confiance = float(request.form["confiance"])

        if confiance < 0 or confiance > 100:
            flash("Le taux de confiance doit être entre 0 et 100 %.")
            return redirect(url_for("accueil"))

        enregistrer_classification(
            materiau_id,
            confiance,
            None,
            "Test manuel"
        )

        flash("Classification enregistrée avec succès !")

    except ValueError:
        flash("Les informations saisies sont invalides.")

    except Exception as erreur:
        flash("Erreur lors de l'enregistrement : " + str(erreur))

    return redirect(url_for("accueil"))


if __name__ == "__main__":
    app.run(debug=True)
