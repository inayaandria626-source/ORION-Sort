
import cv2
import numpy as np


class DetecteurZone:

    def __init__(self):

        self.detecteur = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=32,
            detectShadows=True
        )

        self.nombre_images = 0
        self.images_calibration = 30

    def detecter(self, image_bytes):

        tableau = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            tableau,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError("Image de caméra invalide.")

        hauteur, largeur = image.shape[:2]

        # Réduire la taille pour économiser le processeur
        if largeur > 640:

            nouveau_format = (
                640,
                int(hauteur * 640 / largeur)
            )

            image = cv2.resize(
                image,
                nouveau_format
            )

            hauteur, largeur = image.shape[:2]

        # Préparer l'image
        image_floue = cv2.GaussianBlur(
            image,
            (5, 5),
            0
        )

        # Apprendre l'arrière-plan au démarrage
        if self.nombre_images < self.images_calibration:

            masque = self.detecteur.apply(
                image_floue,
                learningRate=0.05
            )

            self.nombre_images += 1

            return {
                "statut": "calibration",
                "message": "Calibration de la zone...",
                "largeur": largeur,
                "hauteur": hauteur
            }

        # Détecter les changements dans la zone
        masque = self.detecteur.apply(
            image_floue,
            learningRate=0.001
        )

        # Éliminer les ombres
        _, masque = cv2.threshold(
            masque,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # Nettoyer les petits bruits
        noyau = np.ones((5, 5), np.uint8)

        masque = cv2.morphologyEx(
            masque,
            cv2.MORPH_OPEN,
            noyau
        )

        masque = cv2.morphologyEx(
            masque,
            cv2.MORPH_CLOSE,
            noyau
        )

        contours, _ = cv2.findContours(
            masque,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Ignorer les petits changements
        aire_minimale = max(
            1200,
            int(largeur * hauteur * 0.003)
        )

        objets = []

        for contour in contours:

            aire = cv2.contourArea(contour)

            if aire >= aire_minimale:

                x, y, w, h = cv2.boundingRect(contour)

                objets.append({
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "aire": int(aire)
                })

        if not objets:

            return {
                "statut": "aucun_objet",
                "message": "En attente d'un objet.",
                "largeur": largeur,
                "hauteur": hauteur
            }

        # Garder le plus grand objet détecté
        objet = max(
            objets,
            key=lambda element: element["aire"]
        )

        return {
            "statut": "objet_detecte",
            "message": "Objet détecté.",
            "boite": objet,
            "largeur": largeur,
            "hauteur": hauteur
        }


# Créer un seul détecteur pour conserver son arrière-plan
detecteur_zone = DetecteurZone()


def detecter_objet(image_bytes):

    return detecteur_zone.detecter(image_bytes)
