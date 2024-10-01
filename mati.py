import torch
import cv2
import numpy as np
import ssl

# SSL doğrulamasını kapat
ssl._create_default_https_context = ssl._create_unverified_context

# YOLOv5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Kamerayı başlat
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV frame'i YOLO modeline uygun hale getir (BGR'den RGB'ye çeviriyoruz)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Modeli çalıştır ve sonuçları al
    results = model(img)

    # Sonuçları DataFrame olarak al
    df = results.pandas().xyxy[0]

    # Frame'in genişliğini al
    frame_width = frame.shape[1]
    frame_mid_x = frame_width // 2  # Ekran orta noktası (x ekseni)

    # True/False durumunu başlat
    is_mid = False

    # Her bir tespit edilen nesne için döngü
    for index, row in df.iterrows():
        # Koordinatlar
        x_min = int(row['xmin'])
        y_min = int(row['ymin'])
        x_max = int(row['xmax'])
        y_max = int(row['ymax'])

        # Dikdörtgeni çiz
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Dikdörtgenin ortasını hesapla
        rect_mid_x = (x_min + x_max) // 2
        rect_mid_y = (y_min + y_max) // 2

        # Orta noktaya kırmızı nokta çiz
        cv2.circle(frame, (rect_mid_x, rect_mid_y), 5, (0, 0, 255), -1)

        # Eğer dikdörtgenin ortası frame'in ortasına yakınsa "True" yap
        if abs(rect_mid_x - frame_mid_x) < 100:  # 50 pixel yakınlık toleransı
            is_mid = True

    # Yeşil renk için maske oluştur (BGR formatında yeşil aralığı)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])  # Alt yeşil sınır
    upper_green = np.array([80, 255, 255])  # Üst yeşil sınır
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Sadece büyük konturları al
        area = cv2.contourArea(contour)
        if area > 1000:  # Alanı eşikleyerek küçük gürültüleri hariç tutabilirsiniz
            # Konturun etrafına dikdörtgen çiz
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dikdörtgenin köşelerine koordinat yaz
            cv2.putText(frame, f"({x},{y})", (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame, f"({x + w},{y})", (x + w + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame, f"({x},{y + h})", (x - 50, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame, f"({x + w},{y + h})", (x + w + 5, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            # Dikdörtgenin ortasına genişlik ve yükseklik yaz
            cv2.putText(frame, f"W: {w}", (x + w // 2 - 30, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(frame, f"H: {h}", (x - 80, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # True/False mesajını sol üst köşeye yazdır
    if is_mid:
        cv2.putText(frame, "True", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "False", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Sonuçları göster
    cv2.imshow('YOLOv5 Detection', frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak
cap.release()
cv2.destroyAllWindows()
