from flask import Flask, Response, render_template, jsonify
import cv2
import time
from datetime import datetime
import json

from domain.emotion import Emotion
from domain.logger import EmotionLogger
from domain.camera_stream import CameraStream
from services.emotion_detection import EmotionDetectionService
from services.simple_emotion_detector import SimpleEmotionDetector

app = Flask(__name__)

# Global instances
camera = None
camera_available = False
logger = EmotionLogger()

try:
    emotion_detector = EmotionDetectionService()
except Exception as e:
    print(f"AI model yüklenemedi, basit dedektör kullanılıyor: {e}")
    emotion_detector = SimpleEmotionDetector()

# Tespit ayarları
last_detection_time = 0
detection_interval = 3.0  # 3 saniyede bir logla
EMOTION_DETECTION_EVERY_N_FRAMES = 3  # Her 3 frame'de bir duygu analizi yap

def find_camera():
    """Çalışan kamera index'ini bul"""
    print("Kamera aranıyor...")
    for index in range(-1, 10):
        print(f"  Kamera index {index} deneniyor...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"  ✓ Kamera bulundu: index {index}")
                return index
    
    print("  ✗ Hiçbir kamera bulunamadı!")
    return None

def generate_frames():
    """Frame generator - duygu tespiti ile"""
    global camera, emotion_detector, logger, last_detection_time

    print("Video stream başladı...")
    frame_count = 0
    last_emotions: list = []  # Son tespit edilen duygular (frame skip için)

    while True:
        try:
            current_time = time.time()

            frame = camera.read_frame()

            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1
            frame_copy = frame.copy()

            # Yüz tespiti her frame'de
            faces = camera.detect_faces(frame)

            emotions_to_log = []

            if faces:
                # Duygu analizi sadece her N frame'de bir
                run_emotion_detection = (frame_count % EMOTION_DETECTION_EVERY_N_FRAMES == 0)

                if run_emotion_detection:
                    last_emotions = []
                    for face_id, bbox in enumerate(faces, 1):
                        face_roi = camera.extract_face_roi(frame, bbox)
                        emotion_result = emotion_detector.detect_emotion(face_roi)

                        if emotion_result:
                            label, score = emotion_result
                            emotion = Emotion.from_prediction(
                                label=label,
                                score=score,
                                face_id=face_id,
                                bbox=bbox
                            )
                            last_emotions.append((bbox, emotion))

                # Her frame'de en son sonucu çiz
                for bbox, emotion in last_emotions:
                    frame_copy = camera.draw_detection(
                        frame_copy,
                        bbox,
                        emotion.label_tr,
                        emotion.score
                    )
                    emotions_to_log.append(emotion)

            # 3 saniyede bir logla
            if emotions_to_log and (current_time - last_detection_time >= detection_interval):
                logger.log_emotions(emotions_to_log)
                print(f"[LOG] {len(emotions_to_log)} duygu loglandı (Frame: {frame_count})")
                last_detection_time = current_time
            
            # Zaman damgası ekle
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(
                frame_copy,
                timestamp,
                (10, frame_copy.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            
            # JPEG'e çevir
            ret, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # MJPEG formatında gönder
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # FPS kontrolü
            time.sleep(0.033)  # 30 FPS
            
        except Exception as e:
            print(f"Stream hatası: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Sistem durumunu döndür"""
    return jsonify({
        'camera_available': camera_available,
        'message': 'Kamera aktif' if camera_available else 'Kamera bulunamadı veya başlatılamadı'
    })

@app.route('/video_feed')
def video_feed():
    """Video stream endpoint"""
    if not camera_available:
        return Response(status=503)
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/logs')
def get_logs():
    """Tüm logları getir"""
    logs = logger.get_recent_logs(limit=100)
    return jsonify({
        'success': True,
        'logs': logs
    })

@app.route('/api/statistics')
def get_statistics():
    """İstatistikleri getir"""
    stats = logger.get_statistics()
    return jsonify({
        'success': True,
        'statistics': stats
    })

@app.route('/api/clear-logs', methods=['POST'])
def clear_logs():
    """Logları temizle"""
    success = logger.clear_logs()
    return jsonify({
        'success': success,
        'message': 'Loglar temizlendi' if success else 'Hata oluştu'
    })

if __name__ == '__main__':
    try:
        print("=" * 50)
        print("Gerçek Zamanlı Duygu Tanıma Sistemi")
        print("=" * 50)
        print()

        camera_index = find_camera()

        if camera_index is None:
            print("UYARI: Hiçbir kamera bulunamadı. Sunucu yine de başlatılıyor...")
        else:
            camera = CameraStream(camera_index=camera_index)
            if camera.start():
                camera_available = True
                print("✓ Kamera başlatıldı")
            else:
                print("UYARI: Kamera başlatılamadı. Sunucu yine de başlatılıyor...")

        print("\nTarayıcınızda http://localhost:5000 adresini açın")
        print("Durdurmak için Ctrl+C")
        print("=" * 50)
        print()

        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

    except KeyboardInterrupt:
        print("\n\nSistem kapatılıyor...")
    except Exception as e:
        print(f"\nBeklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if camera:
            camera.stop()
        cv2.destroyAllWindows()
        print("Sistem tamamen kapatıldı")