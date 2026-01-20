import customtkinter as ctk
import cv2
from PIL import Image
from ultralytics import YOLO
import supervision as sv
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import os

class SayacApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere Ayarları
        self.title("AI Kişi ve Araç Sayım Paneli")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")

        # Yapay Zeka ve Değişkenler
        self.model = YOLO('yolov8n.pt')
        self.stop_event = threading.Event()
        self.thread = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Yuz algilama hassasiyeti (dusuk scaleFactor + yuksek minNeighbors = daha secici)
        self.face_scale_factor = 1.1
        self.face_min_neighbors = 5
        self.face_min_size = (20, 20)
        
        # Sayaç Hafızası (Excel için)
        self.insan_sayisi = 0
        self.arac_sayisi = 0
        self.motor_sayisi = 0

        self.setup_ui()

    def setup_ui(self):
        # --- Yan Panel (Kontroller) ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="SAYIM KONTROL", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20)

        self.entry_source = ctk.CTkEntry(self.sidebar, placeholder_text="MP4 yolu veya RTSP linki")
        self.entry_source.pack(pady=10, padx=20, fill="x")

        self.btn_baslat = ctk.CTkButton(self.sidebar, text="Analizi Başlat", command=self.start_analysis, fg_color="green")
        self.btn_baslat.pack(pady=10, padx=20, fill="x")

        self.btn_durdur = ctk.CTkButton(self.sidebar, text="Durdur ve Kaydet", command=self.stop_analysis, fg_color="red")
        self.btn_durdur.pack(pady=10, padx=20, fill="x")

        # Sayaç Göstergeleri
        self.lbl_insan = ctk.CTkLabel(self.sidebar, text="İnsan: 0", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_insan.pack(pady=(40, 5))
        
        self.lbl_arac = ctk.CTkLabel(self.sidebar, text="Araç: 0", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_arac.pack(pady=5)

        self.lbl_motor = ctk.CTkLabel(self.sidebar, text="Motorsiklet: 0", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_motor.pack(pady=5)

        # --- Ana Panel (Video Alanı) ---
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Kaynak girin ve Başlat'a basın")
        self.video_label.pack(expand=True, fill="both")

    def start_analysis(self):
        source = self.entry_source.get()
        if not source:
            source = 0  # Boşsa web kamerasını aç
        else:
            # "0" gibi değerleri kamera indeksi olarak kullan
            try:
                source = int(source)
            except ValueError:
                pass
        
        # Sayımları sıfırla
        self.insan_sayisi = 0
        self.arac_sayisi = 0
        self.motor_sayisi = 0
        self.lbl_insan.configure(text="İnsan: 0")
        self.lbl_arac.configure(text="Araç: 0")
        self.lbl_motor.configure(text="Motorsiklet: 0")
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.video_loop, args=(source,), daemon=True)
        self.thread.start()

    def stop_analysis(self):
        self.stop_event.set()
        self.excel_kaydet()

    def excel_kaydet(self):
        # Kaydedilecek veriler
        dosya_adi = "sayim_raporu.xlsx"
        yeni_veri = {
            "Tarih": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Toplam İnsan": [self.insan_sayisi],
            "Toplam Araç": [self.arac_sayisi],
            "Toplam Motorsiklet": [self.motor_sayisi],
            "Kaynak": [self.entry_source.get() if self.entry_source.get() else "Kamera"]
        }
        df = pd.DataFrame(yeni_veri)

        if os.path.exists(dosya_adi):
            mevcut_df = pd.read_excel(dosya_adi)
            df = pd.concat([mevcut_df, df], ignore_index=True)
        
        df.to_excel(dosya_adi, index=False)
        print(f"Rapor {dosya_adi} dosyasına kaydedildi.")

    def video_loop(self, source):
        # Sayılacak Sınıflar (COCO): 0 person, 2 car, 3 motorcycle
        target_person_classes = [0]
        target_vehicle_classes = [2]
        target_motor_classes = [3]

        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.video_label.configure(text="HATA: Kaynağa bağlanılamadı!")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Çizgi Ayarları (Dikey orta çizgi)
        line_start = sv.Point(w // 2, 0)
        line_end = sv.Point(w // 2, h)
        human_counter = sv.LineZone(start=line_start, end=line_end)
        vehicle_counter = sv.LineZone(start=line_start, end=line_end)
        motor_counter = sv.LineZone(start=line_start, end=line_end)
        
        # Görselleştiriciler
        line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        box_annotator = sv.BoxAnnotator()

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_h, frame_w = frame.shape[:2]

            # AI Analizi (Takip Modu)
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                tracker="bytetrack.yaml",
                conf=0.3,
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            if results.boxes.id is not None:
                detections.tracker_id = results.boxes.id.cpu().numpy().astype(int)
            
            # Sadece insanları filtrele
            detections_filtered = detections[[(cid in target_person_classes) for cid in detections.class_id]]

            # Geniş kutulari ikiye bol (kol kola vb. durumlar icin basit ayirma)
            if len(detections_filtered) > 0:
                boxes = detections_filtered.xyxy
                confs = detections_filtered.confidence
                class_ids = detections_filtered.class_id
                tracker_ids = detections_filtered.tracker_id

                split_boxes = []
                split_confs = []
                split_class_ids = []
                split_tracker_ids = [] if tracker_ids is not None else None

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    box_w = x2 - x1
                    box_h = y2 - y1
                    # Kutunun genisligi goruntunun belirli oranindan buyukse bol
                    if box_w > 0.35 * frame_w and box_w / max(box_h, 1.0) > 1.2:
                        mid = (x1 + x2) / 2.0
                        split_boxes.append([x1, y1, mid, y2])
                        split_boxes.append([mid, y1, x2, y2])
                        split_confs.extend([confs[i], confs[i]])
                        split_class_ids.extend([class_ids[i], class_ids[i]])
                        if split_tracker_ids is not None:
                            split_tracker_ids.extend([tracker_ids[i], tracker_ids[i]])
                    else:
                        split_boxes.append([x1, y1, x2, y2])
                        split_confs.append(confs[i])
                        split_class_ids.append(class_ids[i])
                        if split_tracker_ids is not None:
                            split_tracker_ids.append(tracker_ids[i])

                detections_filtered = sv.Detections(
                    xyxy=np.array(split_boxes, dtype=float),
                    confidence=np.array(split_confs, dtype=float),
                    class_id=np.array(split_class_ids, dtype=int),
                    tracker_id=np.array(split_tracker_ids, dtype=int) if split_tracker_ids is not None else None,
                )
            
            # Araç ve motorsiklet filtreleri
            detections_vehicles = detections[[(cid in target_vehicle_classes) for cid in detections.class_id]]
            detections_motors = detections[[(cid in target_motor_classes) for cid in detections.class_id]]

            # Çizgi Geçiş Kontrolü
            human_counter.trigger(detections=detections_filtered)
            vehicle_counter.trigger(detections=detections_vehicles)
            motor_counter.trigger(detections=detections_motors)
            
            # Değerleri Güncelle
            self.insan_sayisi = human_counter.in_count + human_counter.out_count
            self.arac_sayisi = vehicle_counter.in_count + vehicle_counter.out_count
            self.motor_sayisi = motor_counter.in_count + motor_counter.out_count
            
            # Update labels
            self.lbl_insan.configure(text=f"İnsan: {self.insan_sayisi}")
            self.lbl_arac.configure(text=f"Araç: {self.arac_sayisi}")
            self.lbl_motor.configure(text=f"Motorsiklet: {self.motor_sayisi}")
            
            # Yuz kutulari icin basit face detection (person ROI icinde)
            face_boxes = []
            if len(detections_filtered) > 0:
                for (x1, y1, x2, y2) in detections_filtered.xyxy:
                    x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
                    x1_i = max(x1_i, 0)
                    y1_i = max(y1_i, 0)
                    x2_i = min(x2_i, frame_w - 1)
                    y2_i = min(y2_i, frame_h - 1)
                    if x2_i <= x1_i or y2_i <= y1_i:
                        continue
                    roi = frame[y1_i:y2_i, x1_i:x2_i]
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=self.face_scale_factor,
                        minNeighbors=self.face_min_neighbors,
                        minSize=self.face_min_size,
                    )
                    for (fx, fy, fw, fh) in faces:
                        face_boxes.append([x1_i + fx, y1_i + fy, x1_i + fx + fw, y1_i + fy + fh])

            if len(face_boxes) > 0:
                face_detections = sv.Detections(
                    xyxy=np.array(face_boxes, dtype=float),
                    confidence=np.ones(len(face_boxes), dtype=float),
                    class_id=np.zeros(len(face_boxes), dtype=int),
                    tracker_id=None,
                )
                draw_detections = face_detections
            else:
                # Yuz bulunamazsa kisi kutularini goster
                draw_detections = detections_filtered

            # Görüntü Üzerine Çizim (yuz varsa yuzler, yoksa kisiler)
            annotated_frame = box_annotator.annotate(scene=frame, detections=draw_detections)
            line_annotator.annotate(frame=annotated_frame, line_counter=human_counter)

            # Tkinter Arayüzüne Görüntü Aktarma
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((800, 500))
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 500))
            
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk

        cap.release()

if __name__ == "__main__":
    app = SayacApp()
    app.mainloop()
