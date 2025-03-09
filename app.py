import logging
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fer import FER
import uvicorn

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialiser FastAPI
app = FastAPI(
    title="Détection des émotions avec FER",
    description="API pour détecter les émotions dans une image en utilisant le modèle FER.",
    version="1.0.0"
)

# Charger le détecteur FER avec MTCNN
detector = None
try:
    logger.info("🔄 Chargement du modèle FER avec MTCNN...")
    detector = FER(mtcnn=True)
    logger.info("✅ Modèle FER chargé avec succès.")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle : {e}")
    raise RuntimeError("Échec du chargement du modèle. Vérifiez l'installation de `fer`.")

# Endpoint principal
@app.get("/", summary="Message d'accueil")
async def root():
    return {"message": "Bienvenue sur l'API de détection des émotions. Utilisez /predict pour envoyer une image."}

# Endpoint de vérification de l'état
@app.get("/health", summary="Vérification de l'état de l'API")
async def health():
    return {"model_loaded": detector is not None, "status": "healthy"}

@app.post("/predict", summary="Détecter les émotions dans une image")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Accepte une image et renvoie les émotions détectées avec leurs annotations.
    """
    logger.info("📷 Réception d'une image pour analyse...")
    
    # Vérification du format du fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image valide.")

    try:
        # Lire l'image envoyée
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Image invalide")

        # Convertir en RGB pour FER
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détecter les émotions
        logger.info("🔍 Analyse des émotions...")
        result = detector.detect_emotions(frame_rgb)

        # Formater les résultats
        if not result:
            logger.warning("⚠️ Aucun visage détecté dans l'image.")
            return {"message": "Aucun visage détecté", "emotions": []}

        formatted_result = []
        for face in result:
            # Coordonnées du visage
            (x, y, w, h) = face["box"]
            # Landmarks
            landmarks = face.get("keypoints", {})
            # Émotions
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            score = emotions[dominant_emotion]

            formatted_result.append({
                "box": {"x": x, "y": y, "width": w, "height": h},
                "landmarks": {
                    "left_eye": landmarks.get("left_eye", [0, 0]),
                    "right_eye": landmarks.get("right_eye", [0, 0]),
                    "mouth_left": landmarks.get("mouth_left", [0, 0]),
                    "mouth_right": landmarks.get("mouth_right", [0, 0])
                },
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "score": float(score)
            })

        logger.info(f"✅ Émotion dominante détectée : {dominant_emotion}")
        return {"emotions": formatted_result}

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse de l'image : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

# Lancement du serveur
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Utiliser la variable PORT de Render, avec 10000 comme fallback
    logger.info(f"🚀 Lancement de l'API sur le port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)