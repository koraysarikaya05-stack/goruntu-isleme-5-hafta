import cv2
import numpy as np
from matplotlib import pyplot as plt

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'resim.jpg')

# Türkçe karakterli yolları sorunsuz okumak için imdecode kullanımı:
img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
# Görselin başarıyla yüklenip yüklenmediğini kontrol etme
if img is None:
    print("Hata: 'resim.jpg' dosyası okunamadı!")
else:
    # Resmin boyut bilgilerini alma (Yükseklik, Genişlik, Kanal Sayısı)
    height, width, channels = img.shape
    print(f"Resim Boyutları: {width}x{height}, Kanal Sayısı: {channels}")

    # -------------------------------------------------------------
    # 2. Piksellere Doğrudan / Eleman Bazlı Erişim
    # -------------------------------------------------------------
    # Örnek: (y=100, x=50) koordinatındaki piksere ve kanallarına erişim
    ornek_piksel = img[100, 50]
    print(f"(100, 50) piksellik konumun [B, G, R] değerleri: {ornek_piksel}")
    print(f"Mavi Kanal Değeri: {img[100, 50, 0]}")
    print(f"Yeşil Kanal Değeri: {img[100, 50, 1]}")
    print(f"Kırmızı Kanal Değeri: {img[100, 50, 2]}")

    # -------------------------------------------------------------
    # 3. Kanallar İçin Ayrı Ayrı Histogram Hesaplama
    # -------------------------------------------------------------
    # Her renk kanalı için 256 elemanlı frekans dizileri
    hist_b = np.zeros(256, dtype=int)
    hist_g = np.zeros(256, dtype=int)
    hist_r = np.zeros(256, dtype=int)

    # Principle of Spatial Locality (Row-Major Order) gereği:
    # Dış döngü Satır (y), İç döngü Sütun (x) olarak gezilir.
    for y in range(height):
        for x in range(width):
            val_b = img[y, x, 0]  # Mavi kanala erişim
            val_g = img[y, x, 1]  # Yeşil kanala erişim
            val_r = img[y, x, 2]  # Kırmızı kanala erişim

            hist_b[val_b] += 1
            hist_g[val_g] += 1
            hist_r[val_r] += 1

    # -------------------------------------------------------------
    # 4. Histogram Grafiğini Çizdirme
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.title('Her Renk Kanalı İçin Ayrı Ayrı Histogram')
    plt.xlabel('Piksel Yoğunluğu (0 - 255)')
    plt.ylabel('Piksel Frekansı')

    plt.plot(hist_b, color='blue', label='Mavi (Blue) Kanalı')
    plt.plot(hist_g, color='green', label='Yeşil (Green) Kanalı')
    plt.plot(hist_r, color='red', label='Kırmızı (Red) Kanalı')

    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()