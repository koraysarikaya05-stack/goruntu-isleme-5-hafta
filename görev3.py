import cv2
import numpy as np
import os


# 1. KENDİ GRİ SEVİYE DÖNÜŞTÜRME FONKSİYONUMUZ-
def kendi_rgb_to_gray(img_bgr):
    """
    BT.601 Standardı (Y = 0.299*R + 0.587*G + 0.114*B) 
    kullanarak BGR görüntüyü gri tonlu (1 kanallı) görüntüye dönüştürür.
    """
    height, width, _ = img_bgr.shape
    gray_img = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            b = img_bgr[y, x, 0]
            g = img_bgr[y, x, 1]
            r = img_bgr[y, x, 2]
            
            # Formülün uygulanması
            gray_val = 0.299 * r + 0.587 * g + 0.114 * b
            gray_img[y, x] = np.clip(round(gray_val), 0, 255)
            
    return gray_img

# 2. GÖRSELİ OKUMA VE DÖNÜŞÜMLER
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'resim2.jpg')

# Türkçe karakter engelini aşarak okuma
img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

if img is None:
    print("Hata: 'resim.jpg' dosyası okunamadı!")
else:
    # a) OpenCV hazır fonksiyonu ile gri dönüşüm
    gray_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # b) Kendi yazdığımız fonksiyon ile gri dönüşüm
    gray_kendi = kendi_rgb_to_gray(img)

    
    # 3. MUTLAK FARK VE HATA DEĞERİ HESABI
    # Negatif taşmaları önlemek için int32 tipine dönüştürüyoruz
    fark_matrisi = np.abs(gray_opencv.astype(np.int32) - gray_kendi.astype(np.int32))
    toplam_hata = np.sum(fark_matrisi)

    print(f"EŞLEŞEN PİKSELLLERİN MUTLAK FARK TOPLAMI (HATA DEĞERİ): {toplam_hata}")

    # 4. SONUÇLARI OPENCV PENCERESİNDE GÖSTERME (imshow)
  
    cv2.imshow('1. Orijinal Renkli Resim', img)
    cv2.imshow('2. OpenCV cvtColor (Gri)', gray_opencv)
    cv2.imshow('3. Kendi Fonksiyonumuz (Gri)', gray_kendi)

    # Bir tuşa basılana kadar pencereleri açık tutar
    cv2.waitKey(0)
    cv2.destroyAllWindows()