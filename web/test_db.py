
from database import obtenir_connexion

try:
    connexion = obtenir_connexion()

    if connexion.is_connected():
        print("Connexion MySQL réussie !")

    connexion.close()

except Exception as erreur:
    print("Erreur de connexion :", erreur)
