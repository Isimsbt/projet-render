import os
import logging
import cv2
import numpy as np
from fer import FER
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO)

# Initialisation de l'application FastAPI
app = FastAPI()

# Initialisation du détecteur d'émotions
detector = None

# Fonction pour initialiser le détecteur
def initialize_detector():
    global detector
    try:
        detector = FER(mtcnn=True)
        logging.info("✅ Modèle FER chargé avec succès")
    except Exception as e:
        logging.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        raise RuntimeError(f"Erreur lors du chargement du modèle FER: {str(e)}")

# Événements de démarrage et arrêt
@app.on_event("startup")
async def startup_event():
    initialize_detector()

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("🛑 API arrêtée proprement")

# Modèles Pydantic pour la réponse
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

# Route d'accueil
@app.route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Bienvenue sur l'API de détection d'émotions!"}

# Route de santé
@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy"}

# Route de détection d'émotions
@app.post("/detect_emotion", response_model=EmotionResponse)
async def detect_emotion(file: UploadFile = File(...)):
    """Détecte les émotions dans une image téléchargée"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Seules les images sont acceptées")

    try:
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(400, "Impossible de décoder l'image")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_emotions(frame_rgb)

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

# Test avec la webcam (local uniquement)
def test_webcam():
    """Test local avec la webcam - Non exposé via l'API"""
    if detector is None:
        logging.error("Detector is None. Initialization failed.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Impossible d'accéder à la webcam")
        return

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_emotions(frame_rgb)

            for face in results:
                (x, y, w, h) = face['box']
                emotions = face['emotions']
                dominant = max(emotions, key=emotions.get)
                score = emotions[dominant]
                logging.info(f"Frame {frame_count}: Face detected at box ({x}, {y}, {w}, {h})")
                logging.info(f"Frame {frame_count}: Dominant emotion: {dominant} (Score: {score:.1%})")

            frame_count += 1
            if frame_count >= 100:
                break

            import time
            time.sleep(0.1)

    finally:
        cap.release()
        logging.info("Webcam test completed.")

# Point d'entrée principal
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API de détection d'émotions")
    parser.add_argument("--webcam", action="store_true", help="Lancer le test webcam")
    args = parser.parse_args()

    if args.webcam:
        initialize_detector()
        test_webcam()
    else:
        port = int(os.getenv("PORT", 8000))
        uvicorn.run(app, host="0.0.0.0", port=port)