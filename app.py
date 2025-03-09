from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from fer import FER

# Initialiser FastAPI
app = FastAPI(title="Détection des émotions avec FER")

# Ajouter CORS pour permettre les requêtes depuis Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplacez "*" par des domaines spécifiques en production (ex: ["https://votre-app-flutter.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le détecteur FER avec MTCNN
try:
    detector = FER(mtcnn=True)
    print("Modèle FER chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

@app.get("/health", summary="Vérifier la santé de l'API")
async def health_check():
    """
    Endpoint pour vérifier si l'API est opérationnelle et si le modèle est chargé.
    """
    return {"status": "ok", "model_loaded": True}

@app.post("/predict", summary="Détecter les émotions dans une image")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Accepte une image et renvoie les émotions détectées avec leurs annotations.
    """
    try:
        # Vérifier le type de fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")

        # Lire l'image envoyée
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Fichier image vide")

        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Impossible de décoder l'image")

        # Convertir en RGB pour FER
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détecter les émotions
        result = detector.detect_emotions(frame_rgb)

        # Formater les résultats
        if not result:
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

        return {"emotions": formatted_result}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse de l'image : {str(e)}")