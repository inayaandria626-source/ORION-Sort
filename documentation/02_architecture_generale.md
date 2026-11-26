
# ORION-Sort
## Architecture générale du système

### 1. Architecture matérielle

L'ESP32 constitue l'unité de contrôle électronique.
Il communique avec les capteurs et le pilote PCA9685.

Le PCA9685 commande les servomoteurs du bras robotique.

La caméra et l'ordinateur assurent la capture et le traitement des images.

### 2. Architecture logicielle

L'ordinateur exécute plusieurs programmes :

- Python : logique principale.
- OpenCV : traitement des images.
- TensorFlow / Keras : intelligence artificielle.
- MobileNetV2 : classification des déchets.
- Flask : serveur Web.
- MySQL : base de données.

### 3. Communication

L'ESP32 et l'ordinateur communiquent par Wi-Fi sur un réseau local.

L'ordinateur peut envoyer des commandes de mouvement à l'ESP32.

L'ESP32 transmet les informations des capteurs à l'ordinateur.

### 4. Chaîne de traitement

1. Détection de l'objet.
2. Capture de l'image.
3. Classification par IA.
4. Mesure du poids.
5. Détermination du bac.
6. Envoi de la commande au bras.
7. Dépôt de l'objet.
8. Enregistrement des résultats.

### 5. Sécurité et contrôle

Le système doit prévoir :

- Une commande d'arrêt d'urgence.
- Une vérification des données reçues.
- Une gestion des erreurs de classification.
- Une interdiction de mouvement si les conditions de sécurité ne sont pas réunies.
