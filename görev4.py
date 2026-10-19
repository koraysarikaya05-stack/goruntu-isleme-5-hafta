import cv2
import numpy as np
import os

# BGR -> YCrCb RENK UZAYI DÖNÜŞÜM FONKSİYONU
def kendi_bgr_to_ycrcb(img_bgr):
    """
    BGR renk uzayındaki görüntüyü OpenCV standartlarına (BT.601) uygun olarak
    YCrCb (Y, Cr, Cb) renk uzayına dönüştürür.
    """
    height, width, _ = img_bgr.shape
    ycrcb_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # BGR kanallarını int32 tipinde alıyoruz (Taşmaları önlemek için)
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)
    
    # OpenCV BT.601 YCrCb Dönüşüm Katsayıları
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128
    
    # Yuvarlama ve 0-255 aralığına sınırlandırma (OpenCV standart piksel eşlemesi)
    ycrcb_img[:, :, 0] = np.clip(np.round(Y), 0, 255).astype(np.uint8)
    ycrcb_img[:, :, 1] = np.clip(np.round(Cr), 0, 255).astype(np.uint8)
    ycrcb_img[:, :, 2] = np.clip(np.round(Cb), 0, 255).astype(np.uint8)
    
    return ycrcb_img


# GÖRSELİ OKUMA VE KARŞILAŞTIRMA
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'resim2.jpg')

img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

if img is None:
    print("Hata: 'resim2.jpg' dosyası okunamadı!")
else:
    # 1. OpenCV hazır fonksiyonu ile YCrCb dönüşümü
    ycrcb_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 2. Kendi yazdığımız fonksiyon ile YCrCb dönüşümü
    ycrcb_kendi = kendi_bgr_to_ycrcb(img)

    # 3. İki görüntü arasındaki mutlak farkların toplamı (Hata Hesabı)
    fark_matrisi = np.abs(ycrcb_opencv.astype(np.int32) - ycrcb_kendi.astype(np.int32))
    toplam_hata = np.sum(fark_matrisi)

    print(f"OPENCV İLE KENDİ DÖNÜŞÜMÜMÜZ ARASINDAKİ HATA DEĞERİ: {toplam_hata}")

    # Pencerelerde Gösterme
    cv2.imshow('Orijinal BGR', img)
    cv2.imshow('OpenCV YCrCb', ycrcb_opencv)
    cv2.imshow('Kendi YCrCb Dönüşümümüz', ycrcb_kendi)

    cv2.waitKey(0)
    cv2.destroyAllWindows()