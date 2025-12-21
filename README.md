# Gerçek Zamanlı Duygu Tanıma Sistemi

Web kameradan anlık duygu tespiti yapan AI destekli uygulama.

## Özellikler

- 🎭 7 duygu durumu tespit eder (Mutlu, Üzgün, Kızgın, Şaşırmış, Korkmuş, İğrenmiş, Nötr)
- 🎥 Gerçek zamanlı kamera görüntüsü
- 🤖 Vision Transformer (ViT) AI modeli
- 📊 İstatistik ve grafik gösterimi
- 🇹🇷 Türkçe arayüz
- 💾 Otomatik JSON loglama

## Kurulum
```bash
# Sanal ortam oluştur
python -m venv venv

# Aktif et
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Paketleri yükle
pip install -r requirements.txt
```

## Çalıştırma
```bash
python app.py
```

Tarayıcıda aç: `http://localhost:5000`

## Teknolojiler

- Python 3.x
- Flask
- OpenCV
- Transformers (Hugging Face)
- PyTorch
- Chart.js

## Not

İlk çalıştırmada AI modeli indirilir (~500MB). Biraz zaman alabilir.

## Lisans

MIT
