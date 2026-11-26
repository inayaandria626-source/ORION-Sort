
import mysql.connector


def obtenir_connexion():
    connexion = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="orion_sort_db"
    )
    return connexion


# Récupérer les matériaux
def obtenir_materiaux():
    connexion = obtenir_connexion()
    curseur = connexion.cursor(dictionary=True)

    curseur.execute("SELECT * FROM materiaux")
    materiaux = curseur.fetchall()

    curseur.close()
    connexion.close()

    return materiaux


# Calculer les statistiques
def obtenir_statistiques():
    connexion = obtenir_connexion()
    curseur = connexion.cursor(dictionary=True)

    requete = """
        SELECT materiaux.nom, COUNT(classifications.id) AS total
        FROM materiaux
        LEFT JOIN classifications
        ON materiaux.id = classifications.materiau_id
        GROUP BY materiaux.id, materiaux.nom
    """

    curseur.execute(requete)
    resultats = curseur.fetchall()

    curseur.close()
    connexion.close()

    statistiques = {}

    for ligne in resultats:
        statistiques[ligne["nom"].lower()] = ligne["total"]

    statistiques["total"] = sum(statistiques.values())

    return statistiques


# Enregistrer une classification
def enregistrer_classification(materiau_id, confiance,
                               image_path=None, version_modele=None):

    connexion = obtenir_connexion()
    curseur = connexion.cursor()

    requete = """
        INSERT INTO classifications
        (materiau_id, confiance, image_path, version_modele)
        VALUES (%s, %s, %s, %s)
    """

    valeurs = (
        materiau_id,
        confiance,
        image_path,
        version_modele
    )

    curseur.execute(requete, valeurs)

    connexion.commit()

    curseur.close()
    connexion.close()


# Récupérer les 10 dernières classifications
def obtenir_classifications():
    connexion = obtenir_connexion()
    curseur = connexion.cursor(dictionary=True)

    requete = """
        SELECT
            classifications.id,
            materiaux.nom,
            classifications.confiance,
            classifications.version_modele,
            classifications.date_classification
        FROM classifications
        INNER JOIN materiaux
        ON classifications.materiau_id = materiaux.id
        ORDER BY classifications.date_classification DESC
        LIMIT 10
    """

    curseur.execute(requete)
    classifications = curseur.fetchall()

    curseur.close()
    connexion.close()

    return classifications
