# Bu kod, fourier_spectrum.tif dosyasından okunan gri bir görüntüyü kullanarak,
#lojistik dönüşüm uygular ve bu işlem sonrası orijinal ve dönüştürülmüş görüntüleri yatay olarak birleştirerek ekranda görüntüler.
#Lojistik dönüşüm, genellikle sınıflandırma algoritmalarında kullanılan bir matematiksel dönüşüm işlemidir. Bu işlemde bir değişkenin değeri, sigmoid fonksiyonu adı verilen bir fonksiyonla yeniden ölçeklendirilir. Sigmoid fonksiyonu, girdi olarak aldığı herhangi bir sayıyı 0 ile 1 arasında bir değere dönüştürür.
#Bu işlem, bir veri kümesindeki sayısal değişkenlerin değerlerini, algoritmaların daha iyi çalışması için uygun bir aralığa getirir. Ayrıca lojistik dönüşüm, lojistik regresyon gibi algoritmaların temelinde de kullanılır.
#img_path değişkenine fourier_spectrum.tif dosyasının yolunu atar.
import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path = "./img/fourier_spectrum.tif"
#cv2.imread verilen dosya yolu üzerinden resmi okur. İkkinci argüman olarak 0 verilerek, görüntünün gri ölçekte okunmasını sağlar.
img = cv2.imread(img_path, 0)
#log_trans işlevi, verilen bir görüntüde lojistik dönüşüm gerçekleştirir. Dönüşümün belirlenmesi için 'c' parametresi kullanılır. Bu işlevde, öncelikle görüntü float64 veri türüne dönüştürülür ve ardından lojistik dönüşüm uygulanır. 'img_scale()' işlevi, görüntüdeki piksel değerlerini 0 ile 255 arasında scale eder.
def log_trans(img, c):
    img = img.astype(np.float64)
    s = c * np.log(1 + img)
    s = img_scale(s)
    return s
#img_scale işlevi, görüntüdeki piksel değerlerini 0 ile 255 arasına scale etmek için kullanılır. İşlev, görüntünün minimum değerini çıkarır, ardından görüntünün maksimum değerine bölerek görüntüyü 0 ile 1 arasına scale eder. Daha sonra, görüntüyü 255 ile çarparak ve uint8 veri tipine dönüştürerek ölçeklendirir.
def img_scale(img):
    img -= np.min(img)
    img /= np.max(img)
    img *= 255
    return img.astype(np.uint8)
#uint8 den float64 dönüşümü
#deneme adlı NumPy dizisi, np.array işleviyle oluşturulur. Dizi, uint8 veri tipinde 1, 25, 126, 250, 255 ve 254 değerlerini içerir. Dizinin elemanları daha sonra float64 veri tipine dönüştürülür ve 2 ile toplanır. Sonuç olara, deneme dizisi şu anda [3.0,27.0,128.0,252.0,257.0,256.0] elemenlarına sahiptir.
deneme = np.array([1,25,126,250, 255, 254], dtype=np.uint8)
deneme =deneme.astype(np.float64)
deneme = 2 + deneme
print(deneme)

img_log = log_trans(img, c=1)
print(np.min(img))
print(np.max(img))
print(np.min(img_log))
print(np.max(img_log))
# # belirli aralıktaki değerleri 0 ile 255 arasına scale ettik
deneme = np.array([2.3, 2.5, 2.8, 3.5,3.7, 4.6, 5.9], dtype=np.float64)
deneme = deneme - np.min(deneme)
deneme = deneme / np.max(deneme)
deneme = deneme * 255
deneme = deneme.astype(np.uint8)
#np.min() ve np.max() işlevleri, verilen dizi veya görüntünün minimum ve maksimum piksel değerlerini hesaplar.
print(deneme)
print(np.min(deneme))
print(np.max(deneme))
#img_log adlı değişkene log_trans() işlevi kullanılarak, görüntü üzerinde lojistik dönüşüm uygulanır. 'c' parametresi burada 1 olarak ayarlanmıştır.
img_log = log_trans(img, c=1)
#hstacked adlı değişkene, yatay olarak iki görüntü birleştirilir.
hstacked = np.hstack((img, img_log))
plt.imshow(hstacked, cmap="gray")
plt.show()
