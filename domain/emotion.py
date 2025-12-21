from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Emotion:
    """Duygu durumu domain entity"""
    
    # Duygu çevirileri
    EMOTION_TR = {
        'angry': 'Kızgın',
        'disgust': 'İğrenmiş',
        'fear': 'Korkmuş',
        'happy': 'Mutlu',
        'sad': 'Üzgün',
        'surprise': 'Şaşırmış',
        'neutral': 'Nötr'
    }
    
    label: str
    score: float
    timestamp: datetime
    face_id: int
    bbox: tuple  # (x, y, width, height)
    
    @property
    def label_tr(self) -> str:
        """Türkçe duygu etiketi"""
        return self.EMOTION_TR.get(self.label.lower(), self.label)
    
    def to_dict(self) -> dict:
        """JSON formatına dönüştür"""
        return {
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'face_id': int(self.face_id),
            'emotion': self.label_tr,
            'emotion_en': self.label,
            'score': round(float(self.score), 2),
            'bbox': {
                'x': int(self.bbox[0]),
                'y': int(self.bbox[1]),
                'width': int(self.bbox[2]),
                'height': int(self.bbox[3])
            }
        }
    
    @classmethod
    def from_prediction(cls, label: str, score: float, face_id: int, bbox: tuple) -> 'Emotion':
        """Tahmin sonucundan Emotion nesnesi oluştur"""
        return cls(
            label=label,
            score=float(score),
            timestamp=datetime.now(),
            face_id=int(face_id),
            bbox=tuple(int(x) for x in bbox)
        )