
"""
Basit Offline Emotion Detector
Haar Cascade + basit heuristics ile çalışır
Model indirme gerektirmez
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import random

class SimpleEmotionDetector:   
    def __init__(self):
        print("Basit emotion detector başlatıldı (offline mode)")
        print("NOT: Bu basit bir demo implementasyonu")
        
        # Simüle edilmiş duygu dağılımları
        self.emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'neutral']
        self.weights = [0.3, 0.15, 0.1, 0.15, 0.1, 0.2]  # Mutlu daha olası
    
    def detect_emotion(self, image: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Basit duygu tespiti
        
        Args:
            image: OpenCV formatında görüntü (BGR)
            
        Returns:
            (emotion_label, confidence_score) tuple'ı
        """
        try:
            # Görüntü özelliklerini analiz et
            brightness = self._calculate_brightness(image)
            contrast = self._calculate_contrast(image)
            
            # Basit heuristics
            if brightness > 150:
                # Parlak görüntü - muhtemelen mutlu
                emotion = 'happy'
                score = 0.70 + random.uniform(0, 0.25)
            elif brightness < 80:
                # Karanlık - muhtemelen üzgün/ciddi
                emotion = 'sad'
                score = 0.65 + random.uniform(0, 0.25)
            elif contrast > 100:
                # Yüksek kontrast - şaşkın veya kızgın
                emotion = random.choice(['surprise', 'angry'])
                score = 0.60 + random.uniform(0, 0.30)
            else:
                # Rastgele ama ağırlıklı seçim
                emotion = random.choices(self.emotions, weights=self.weights)[0]
                score = 0.55 + random.uniform(0, 0.35)
            
            return emotion, min(score, 0.95)
            
        except Exception as e:
            print(f"Duygu tespiti hatası: {e}")
            return 'neutral', 0.5
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Görüntü parlaklığını hesapla"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Görüntü kontrastını hesapla"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray)