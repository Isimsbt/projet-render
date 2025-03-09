import os
import logging
import cv2
import numpy as np
from fer import FER
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import imghdr
import uvicorn

# Configuration initiale
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppression des warnings TensorFlow
logging.basicConfig(level=logging.INFO)

app = FastAPI()

detector = None

# Événements de démarrage et arrêt
async def startup_event():
    global detector
    try:
        detector = FER(mtcnn=True)
        logging.info("✅ Modèle FER chargé avec succès")
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        raise

async def shutdown_event():
    pass

app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# Modèle Pydantic pour la réponse
class EmotionBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class FaceEmotion(BaseModel):
    box: EmotionBox
    dominant_emotion: str
    score: float
    all_emotions: dict

class EmotionResponse(BaseModel):
    emotions: List[FaceEmotion]
    message: Optional[str] = None

# Routes
@app.get("/", response_model=dict)
async def root():
    return {"message": "Bienvenue sur l'API de détection d'émotions!"}

@app.post("/detect_emotion", response_model=EmotionResponse)
async def detect_emotion(file: UploadFile = File(...)):
    """Détecte les émotions dans une image téléchargée"""
    # Validation du fichier image
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Fichier non supporté - Uniquement les images sont acceptées")

    try:
        # Vérification du format de l'image
        image_data = await file.read()
        image_type = imghdr.what(None, h=image_data)
        if not image_type:
            raise HTTPException(400, "Format d'image invalide")

        # Conversion en tableau numpy
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Impossible de décoder l'image")

        # Détection des émotions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_emotions(frame_rgb)

        # Formatage de la réponse
        emotions = []
        for face in results:
            box = EmotionBox(**dict(zip(['x', 'y', 'width', 'height'], face['box'])))
            emotions_dict = face['emotions']
            dominant = max(emotions_dict, key=emotions_dict.get)
            emotions.append(FaceEmotion(
                box=box,
                dominant_emotion=dominant,
                score=emotions_dict[dominant],
                all_emotions=emotions_dict
            ))

        return EmotionResponse(emotions=emotions, message="Détection réussie")

    except Exception as e:
        logging.error(f"Erreur lors de la détection : {str(e)}", exc_info=True)
        raise HTTPException(500, "Erreur interne du serveur")

# Fonction de test avec webcam (non exposée via l'API)
def test_webcam():
    """Test local avec la webcam - Non exposé via l'API"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Impossible d'accéder à la webcam")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_emotions(frame_rgb)

            for face in results:
                (x, y, w, h) = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                emotions = face['emotions']
                dominant = max(emotions, key=emotions.get)
                score = emotions[dominant]

                label = f"{dominant} ({score:.1%})"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            cv2.imshow('Détection en temps réel (Appuyez sur Q pour quitter)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Point d'entrée principal
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API de détection d'émotions")
    parser.add_argument("--webcam", action="store_true", help="Lancer le test webcam")
    args = parser.parse_args()

    if args.webcam:
        test_webcam()
    else:
        # Exécution locale sur localhost:8000
        uvicorn.run(app, host="localhost", port=8000)