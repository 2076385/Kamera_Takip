# Kamera_Takip
Yerel kamera veya video kaynagi uzerinden kisi sayim uygulamasi.

## Kurulum
1) Depoyu klonla:
```bash
git clone https://github.com/<kullanici>/<repo>.git
cd <repo>
```

2) Sanal ortam olustur ve bagimliiklari yukle:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3) Calistir:
```bash
python main.py
```

## Kullanim
- Uygulamada kaynak alani bos birakilirsa varsayilan kamera (0) acilir.
- Baska kamera icin kaynak alanina 1, 2 gibi kamera indeksleri girilebilir.
- MP4 dosyasi veya RTSP linki de girilebilir.

## Notlar
- `yolov8n.pt` modeli ilk calistirmada otomatik indirilebilir (internet gerekir).
- Her bilgisayarda kendi kamerasini kullanmak icin bu kurulum adimlari aynen uygulanir.
