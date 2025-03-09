from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from fer import FER

# Initialiser FastAPI
app = FastAPI(title="Détection des émotions avec FER")

# Charger le détecteur FER avec MTCNN
try:
    detector = FER(mtcnn=True)
    print("Modèle FER chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    exit()

@app.post("/predict", summary="Détecter les émotions dans une image")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Accepte une image et renvoie les émotions détectées avec leurs annotations.
    """
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")