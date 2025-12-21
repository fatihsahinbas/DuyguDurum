import cv2
import numpy as np
from typing import Optional, List, Tuple
from threading import Lock
import time
from PIL import Image, ImageDraw, ImageFont

class CameraStream:
    """Kamera akışı domain entity"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_lock = Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.is_running = False
        
        # Türkçe font için PIL kullanacağız
        try:
            # Windows için
            self.font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                # Linux için
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                # Varsayılan font
                self.font = ImageFont.load_default()
    
    def start(self) -> bool:
        """Kamerayı başlat"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            # Kamera ayarları
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            return True
        except Exception as e:
            print(f"Kamera başlatma hatası: {e}")
            return False
    
    def stop(self) -> None:
        """Kamerayı TAMAMEN durdur"""
        print("Kamera stop() çağrıldı...")
        self.is_running = False
        
        if self.cap:
            # Birkaç kere release dene
            for i in range(3):
                self.cap.release()
                time.sleep(0.1)
            
            self.cap = None
        
        # Tüm OpenCV pencerelerini kapat
        cv2.destroyAllWindows()
        
        # Biraz bekle ki kamera ışığı sönebilsin
        time.sleep(0.5)
        
        print("Kamera kaynakları serbest bırakıldı")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Bir frame oku"""
        if not self.cap or not self.is_running:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            with self.frame_lock:
                self.current_frame = frame.copy()
            return frame
        return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Mevcut frame'i getir"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Frame'de yüz tespiti yap"""
        # Haar Cascade ile yüz tespiti
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [tuple(face) for face in faces]
    
    def extract_face_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Yüz bölgesini çıkar"""
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w]
        return face_roi
    
    def put_turkish_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                         font_size: int = 20, color: Tuple[int, int, int] = (255, 255, 255),
                         bg_color: Tuple[int, int, int] = (102, 0, 153)) -> np.ndarray:
        """Türkçe karakter destekli metin yaz"""
        try:
            # OpenCV BGR'den RGB'ye
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            draw = ImageDraw.Draw(pil_image)
            
            # Font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Metin boyutunu al
            bbox = draw.textbbox(position, text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Arka plan dikdörtgeni
            bg_position = [
                position[0] - 5,
                position[1] - text_height - 5,
                position[0] + text_width + 5,
                position[1] + 5
            ]
            draw.rectangle(bg_position, fill=bg_color)
            
            # Metni yaz
            draw.text(position, text, font=font, fill=color)
            
            # PIL'den OpenCV'ye geri dön
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            print(f"Türkçe metin yazma hatası: {e}")
            # Hata olursa normal cv2.putText kullan
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return frame
    
    def draw_detection(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        label: str, 
        score: float,
        color: Tuple[int, int, int] = (102, 0, 153)
    ) -> np.ndarray:
        """Tespit edilen yüz ve duyguyu frame üzerine çiz - TÜRKÇE DESTEKLI"""
        x, y, w, h = bbox
        
        # Dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Türkçe metin
        text = f"{label} ({score:.2f})"
        text_position = (x, y - 10)
        
        # PIL ile Türkçe karakterli metin yaz
        frame = self.put_turkish_text(
            frame, 
            text, 
            text_position, 
            font_size=18,
            color=(255, 255, 255),
            bg_color=color
        )
        
        return frame
    
    def __del__(self):
        """Cleanup - kamera kapatılsın"""
        self.stop()