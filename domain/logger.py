import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter
from domain.emotion import Emotion

MAX_LOG_ENTRIES = 5000  # Dosyada tutulacak maksimum kayıt sayısı

class EmotionLogger:
    """Duygu durumu loglama domain service"""

    def __init__(self, log_file: str = 'emotion_logs.json'):
        self.log_file = Path(log_file)
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Log dosyasının varlığını kontrol et"""
        if not self.log_file.exists():
            self.log_file.write_text('[]', encoding='utf-8')
    
    def log_emotion(self, emotion: Emotion) -> None:
        """Duygu durumunu logla"""
        logs = self._read_logs()
        logs.append(emotion.to_dict())
        self._write_logs(logs[-MAX_LOG_ENTRIES:])

    def log_emotions(self, emotions: List[Emotion]) -> None:
        """Birden fazla duygu durumunu logla"""
        if not emotions:
            return

        logs = self._read_logs()
        logs.extend([e.to_dict() for e in emotions])
        self._write_logs(logs[-MAX_LOG_ENTRIES:])
    
    def get_all_logs(self) -> List[Dict[str, Any]]:
        """Tüm logları getir"""
        return self._read_logs()
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Son N adet logu getir"""
        logs = self._read_logs()
        return logs[-limit:]
    
    def clear_logs(self) -> bool:
        """Tüm logları temizle"""
        try:
            self._write_logs([])
            return True
        except Exception as e:
            print(f"Log temizleme hatası: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Duygu istatistiklerini hesapla"""
        logs = self._read_logs()
        
        if not logs:
            return {
                'total_detections': 0,
                'emotion_counts': {},
                'average_scores': {},
                'most_common': None,
                'least_common': None
            }
        
        # Duygu sayıları
        emotions = [log['emotion'] for log in logs]
        emotion_counter = Counter(emotions)
        
        # Ortalama skorlar
        emotion_scores = {}
        for log in logs:
            emotion = log['emotion']
            if emotion not in emotion_scores:
                emotion_scores[emotion] = []
            emotion_scores[emotion].append(log['score'])
        
        avg_scores = {
            emotion: round(sum(scores) / len(scores), 2)
            for emotion, scores in emotion_scores.items()
        }
        
        most_common = emotion_counter.most_common(1)
        least_common = emotion_counter.most_common()[-1] if emotion_counter else None
        
        return {
            'total_detections': len(logs),
            'emotion_counts': dict(emotion_counter),
            'average_scores': avg_scores,
            'most_common': {
                'emotion': most_common[0][0],
                'count': most_common[0][1]
            } if most_common else None,
            'least_common': {
                'emotion': least_common[0],
                'count': least_common[1]
            } if least_common else None
        }
    
    def _read_logs(self) -> List[Dict[str, Any]]:
        """Log dosyasını oku"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_logs(self, logs: List[Dict[str, Any]]) -> None:
        """Log dosyasına yaz"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)