import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 1. KENDİ CALCHIST FONKSİYONUMUZ

def kendi_calc_hist(kanal):
    """
    OpenCV'nin cv2.calcHist fonksiyonunu taklit eder.
    Tek bir renk kanalındaki piksel frekanslarını hesaplar.
    """
    hist = np.zeros(256, dtype=int)
    height, width = kanal.shape
    
    # Principle of Spatial Locality (Row-Major Order)
    for y in range(height):
        for x in range(width):
            piksel_degeri = kanal[y, x]
            hist[piksel_degeri] += 1
            
    return hist

# 2. GÖRSELİ OKUMA VE İŞLEME

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'resim2.jpg')

# Türkçe karakter engelini aşarak okuma
img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

if img is None:
    print("Hata: 'resim2.jpg' dosyası okunamadı!")
else:
    # BGR Kanallarını ayırma
    blue_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    red_channel = img[:, :, 2]

    # Fonksiyonu her kanal için ayrı ayrı çalıştırma
    hist_b = kendi_calc_hist(blue_channel)
    hist_g = kendi_calc_hist(green_channel)
    hist_r = kendi_calc_hist(red_channel)

 
    # 3. HISTOGRAM GRAFİĞİNİ ÇİZDİRME
    plt.figure(figsize=(10, 6))
    plt.title('Özel calcHist Fonksiyonu İle Kanal Histogramları')
    plt.xlabel('Piksel Yoğunluğu (0 - 255)')
    plt.ylabel('Piksel Frekansı')

    plt.plot(hist_b, color='blue', label='Mavi (Blue) Kanalı')
    plt.plot(hist_g, color='green', label='Yeşil (Green) Kanalı')
    plt.plot(hist_r, color='red', label='Kırmızı (Red) Kanalı')

    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()