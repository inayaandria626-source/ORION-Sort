
# ORION-Sort
## Cahier des charges

### 1. Présentation du projet

ORION-Sort est un système intelligent de tri automatique des déchets utilisant un bras robotique et la vision artificielle.

Son objectif est de reconnaître différents types de déchets et de les déposer dans des poubelles adaptées.

### 2. Problématique

Le tri manuel des déchets demande du temps et peut entraîner des erreurs de classification.

ORION-Sort cherche à automatiser cette opération grâce à l'intelligence artificielle, aux capteurs et à la robotique.

### 3. Objectifs

- Détecter la présence d'un déchet.
- Capturer une image de l'objet.
- Identifier sa matière avec l'intelligence artificielle.
- Mesurer son poids.
- Saisir l'objet avec un bras robotique.
- Déposer l'objet dans le bac correspondant.
- Enregistrer les résultats dans une application Web.

### 4. Types de déchets

Le système doit reconnaître :

- Plastique
- Métal
- Papier
- Carton
- Vide (aucun objet détecté)

### 5. Architecture générale

Le système comporte cinq modules :

1. Électronique : ESP32 et capteurs.
2. Robotique : bras articulé et pince.
3. Intelligence artificielle : reconnaissance des déchets.
4. Mathématiques : calculs et précision des mouvements.
5. Web : tableau de bord et historique du tri.

### 6. Fonctionnement général

1. Le capteur détecte un objet.
2. La caméra capture son image.
3. Le modèle IA identifie sa matière.
4. La balance mesure son poids.
5. Le système détermine le bac de destination.
6. Le bras saisit et transporte l'objet.
7. L'objet est déposé dans le bac adapté.
8. Les informations sont enregistrées dans le tableau de bord.

### 7. Résultats attendus

- Reconnaissance correcte des déchets.
- Tri automatique fiable.
- Mouvements précis du bras robotique.
- Historique des opérations.
- Statistiques par type de déchet.
- Possibilité de commande manuelle depuis l'interface Web.

### 8. Technologies envisagées

- ESP32
- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- Flask
- MySQL
- HTML, CSS et JavaScript
