from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

class EmotionDetectionService:
    def __init__(self):
        print("AI Model yükleniyor... (ilk kez ~500MB indirecek)")
        self.pipe = pipeline(
            "image-classification",
            model="trpakov/vit-face-expression"
        )
        print("✓ AI Model hazır!")
    
    def detect_emotion(self, image: np.ndarray):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        results = self.pipe(pil_image)
        
        if results:
            best = results[0]
            return best['label'], best['score']
        return None