#Bu kod, verilen görüntüye kuvvet dönüşümü yaparak, farklı gama(y) değerleri için üç farklı sonuç görüntüsü oluşturur ve bunları yatay ve dikey olarak birleştirerek ekrana yansıtır.
#İlk olarak "kuvvet_donusumu" fonksiyonu, bir görüntü(r), bir sabit(c) ve bir kuvvet(y) alarak kuvvet dönüşümünü hesaplar ve bu işlem sonucu oluşan görüntüyü "rescale" fonksiyonunu kullanarak 0 ile 255 arasına ölçekler. 
#Daha sonra, MR ve şehir görüntüleri farklı y değerleri ile kuvvet dönüşümüne tabi tutulur ve sonuçlar ayrı ayrı oluşturulur. Son olarak yatay ve dikey olarak birleştirilerek ekrana yansıtılır.

#𝑠 =𝑐𝑟^𝛾
#kuvvet_donusumu fonksiyonu, bir girdi goruntusu(r), sabir bir çarpan(c) ve bir üs(gamma) alarak, her piksel değerini 𝑠 =𝑐𝑟^𝛾 formülüne göre yeniden hesaplar. Bu işlemi yaparken görüntüyü önce np.float64 türüne dönüştürür ve sonra "rescale" adlı başka bir fonksiyon kullanarak yeniden ölçeklendirir.
#Sonuç olarak, işlemden elde edilen görüntü(s) döndürülür.
import cv2
import numpy as np
import matplotlib.pyplot as plt

def kuvvet_donusumu(r, c, gamma):
    r = r.astype(np.float64)
    s = c * r ** gamma
    s = rescale(s)
    return s
#"rescale" fonksiyonu, ölçeklendirilecek bir görüntü(img) alarak, piksel değerlerini 0 ile 255 arasına ölçeklendirir. Bu amaçla öncelikle img'den minimum değeri çıkararak piksel değerlerini sıfırdan başlatır ve sonrasında img'yi maksimum değere böler böylece tüm piksel değerleri 0 ve 1 arasında bir değere düşer.
#Son olarak, piksel değerlerini 255 ile çarpıp np.uint8 veri türüne dönüştürerek 0-255 arasındaki piksel değerlerine getirir.
def rescale(img):
    img -= np.min(img) # 2-7 -- > 0 - 5  // 0 ile x arasına getiriyoum
    img /= np.max(img) # 0 - 1 arasında değerler elde edeceğiz
    img *=255 # 0 - 255 arasına değerlerimizi getirdik
    return img.astype(np.uint8)
#Bir mr görüntüsü yüklenir. 
mr_path = "./img/mri.tif"
mr_img = cv2.imread(mr_path, 0)
#c adında bir sabir ve ör_gammas adlı bir dizi belirlenerek, bu sabiti ve her bir gamma değerini "kuvvet_donusumu" fonksiyonuna vererek yeni görüntüler oluşturur.
#Bu yeni görüntüşeri mr_images listesinde saklar ve sonrasında np.hstack ve np.vstack işlemlerini kullanarak bu görüntüleri birleştirir
c=1
mr_gammas = [0.6, 0.4, 0.3]
mr_images = []
for mr_gamma in mr_gammas:
    donusen = kuvvet_donusumu(mr_img, c, mr_gamma)
    mr_images.append(donusen)
mr_hstacked_ust = np.hstack((mr_img, mr_images[0]))
mr_hstacked_alt = np.hstack((mr_images[1], mr_images[2]))
mr_vstacked = np.vstack((mr_hstacked_ust, mr_hstacked_alt))
#sonuç görüntüsü çizdirilir
plt.imshow(mr_vstacked, cmap="gray")
plt.show() #Bu kod MR görüntüsüne uygulanan kuvvet dönüşümü işleminin farklı gamma değerleri için nasıl farklı sonuçlar ürettiğini gösterir.


#görüntü dosyası okunur ve sehir_img değişkenine atanır.
sehir_path = "./img/sehir.tif"
sehir_img = cv2.imread(sehir_path, 0)
#sehir_gammas adlı bir dizi oluşturur ve bu dizide[3,4,5] değerleri bulunur.
sehir_gammas = [3, 4, 5]
#sehir_images adlı boş liste oluşturulur.
sehir_images = []
#Bir döngü oluşturulur ve bu döngü "sehir_gammas" dizisindeki her bir değer için tekrarlanır. Döngünün her turunda "sehir_gamma" adlı değişken, "sehir_gammas" dizisidindeki sıradaki değere eşitlenir
#"kuvvet_donusumu" fonksiyonu "sehir_img","c" ve "sehir_gamma" değişkenlerini kullanarak "donusen" adlı yeni bir görüntü oluşturur. 
#"donusen" adlı görüntü, "sehir_images" adlı listeye eklenir.
for sehir_gamma in sehir_gammas:
    donusen = kuvvet_donusumu(sehir_img,c,sehir_gamma)
    sehir_images.append(donusen)
#"np.hstack" ve "np.vstack" fonksiyonları kullanılarak "sehir_img" ve "sehir_images" listesindeki görüntüler yatay ve dikey olarak birleştirilir.
sehir_hstacked1 = np.hstack((sehir_img, sehir_images[0]))
sehir_hstacked2 = np.hstack((sehir_images[1], sehir_images[2]))
sehir_vstacked = np.vstack((sehir_hstacked1, sehir_hstacked2))
#Elde edilen birleştirilmiş görüntü "plt.imshow()" fonksiyonu kullanılarak gri tonlamalı olarak gösterilir.
plt.imshow(sehir_vstacked, cmap="gray")
plt.show()
