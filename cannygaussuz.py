import cv2
import numpy as np
import os

# 1. Sobel Filtre ile Gradyan Hesaplama
def sobel_gradients(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    Ix = cv2.filter2D(img, -1, Kx)
    Iy = cv2.filter2D(img, -1, Ky)
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255 if G.max() > 0 else G
    theta = np.arctan2(Iy, Ix)
    return G, theta

# 2. Non-Maximum Suppression (Kenar İnceltme)
def non_max_suppression(G, theta):
    h, w = G.shape
    Z = np.zeros((h, w), dtype=np.float32)
    angle = np.rad2deg(theta) % 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            ang = angle[i, j]
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q, r = G[i, j+1], G[i, j-1]
            elif 22.5 <= ang < 67.5:
                q, r = G[i+1, j-1], G[i-1, j+1]
            elif 67.5 <= ang < 112.5:
                q, r = G[i+1, j], G[i-1, j]
            elif 112.5 <= ang < 157.5:
                q, r = G[i-1, j-1], G[i+1, j+1]

            if G[i, j] >= q and G[i, j] >= r:
                Z[i, j] = G[i, j]
    return Z

# 3. Hysteresis Thresholding (Çift Eşikleme)
def hysteresis_thresholding(img, low_ratio=0.05, high_ratio=0.15):
    high_thresh = img.max() * high_ratio
    low_thresh = img.max() * low_ratio
    
    res = np.zeros_like(img, dtype=np.uint8)
    strong, weak = 255, 50

    strong_i, strong_j = np.where(img >= high_thresh)
    weak_i, weak_j = np.where((img >= low_thresh) & (img <= high_thresh))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if res[i, j] == weak:
                if (res[i-1:i+2, j-1:j+2] == strong).any():
                    res[i, j] = strong
                else:
                    res[i, j] = 0
    return res

# Ana Çağrı Fonksiyonu (Gauss Yumuşatma Kaldırıldı)
def custom_canny_no_blur(image_path):
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"HATA: '{image_path}' dosyası bulunamadı veya okunamadı!")
        return None

    # Gauss filtresi atlanıp doğrudan orijinal resim kullanılıyoruz
    G, theta = sobel_gradients(img)
    nms = non_max_suppression(G, theta)
    edges = hysteresis_thresholding(nms)
    return edges

# Ana çalıştırma bloğu
if __name__ == "__main__":
    script_dir = r"C:\Users\Casper\Desktop\görüntü işleme\5.hafta"
    
    image_path = os.path.join(script_dir, "resim.jpg")
    output_path = os.path.join(script_dir, "cannygaussyok_sonuc.png")
    
    print("İşlem başlatılıyor (Gauss yumuşatma olmadan)...")
    result = custom_canny_no_blur(image_path)
    
    if result is not None:
        is_success, buffer = cv2.imencode(".png", result)
        if is_success:
            buffer.tofile(output_path)
            print(f"BAŞARILI: Görsel kaydedildi -> {output_path}")
        else:
            print("HATA: Görsel encode edilemedi!")

        cv2.imshow("Canny Kenar Tespiti Sonucu (Gauss Olmadan)", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("HATA: Görsel okunamadı veya işlenemedi!")