import cv2
import numpy as np
import matplotlib.pyplot as plt
#bir resim dosyası okunur ve "img" değişkenine atanır
img_path="./images/manzara.jpg"
img=cv2.imread(img_path)
#"img.shape" komutu, okunan resmin boyutlarını(yukseklik, genislik, kanal sayısı) gösterir.
print(img.shape)
#np.min komutu görüntünün en küçük piksel değerini gosterir.
print("Minimum",np.min(img))
#np.max komutu görüntünün en büyük piksel değerini gösterir.
print("Maximum",np.max(img))
#img[0,0,0], img[0,0,1], img[0,0,2] komutları, sırasıyla mavi, yeşil ve kırmızı renk kanallarındaki (RGB) ilk pikselin değerlerini gösterir.
print("Blue Kanali:",img[0,0,0]) #blue kanalı için 0
print("Green Kanali:",img[0,0,1]) #green kanalı için 1
print("Red Kanali:", img[0,0,2]) #red kanalı için 2
#bas_satir, bit_satir, bas_sutun, bit_sutun değişkenleri, görüntünün belirli bir bölgesini kesmek için kullanılır.
bas_satir=215
bit_satir=300
bas_sutun=230
bit_sutun=350
#"img_kes" değişkeni, belirtilen koordinatlardaki bölgeyi keser ve yeni bir görüntü oluşturur.
img_kes=img[bas_satir:bit_satir,bas_sutun:bit_sutun] 
#cv2.imshow ve cv2.waitKey komutları, kesilen görüntüyü görüntüler.
cv2.imshow("kesilen resim",img_kes)
cv2.waitKey(0)
cv2.destroyAllWindows()
#img_kes_R, img_kes_G, img_kes_B değişkenleri, kesilen görüntünün sırasıyla kırmızı, yeşil ve mavi kanallarındaki piksel değerlerini içerir.
img_kes_R=img_kes[:,:,2]
img_kes_G=img_kes[:,:,1]
img_kes_B=img_kes[:,:,0]
#Aşağıdaki kod, aynı resmin gri tonlamalı sürümünü yükler ve "img_grayscale" değişkenine atar. "img_grayscale.shape" komutu, gri tonlamalı görüntünün boyutlarını gösterir
img_grayscale=cv2.imread(img_path,0)
print(img_grayscale.shape)
#"img_grayscale[0,0]" komutu, görüntünün ilk pikselinin gri tonlamalı değerini gösterir.
print(img_grayscale[0,0])


# img_path değişkenine mamogram.png dosyasının yolu atanır
img_path = "./images/mamogram.png"
# cv2.imread işlevi, verilen dosya yolu üzerinden resmi okur. İkinci argüman olarak 0 verilerek, görüntünün gri ölçekte okunmasını sağlar.
img = cv2.imread(img_path, 0)
# görüntünün boyutları yazdırılır. Burada (h,w) formunda çıktı verecektir.
print(img.shape)
# s = L - 1 - r
# 1-256 --> 0-255
# s = negatifi alınmış piksel
# r = grayscale resimdeki her bir pikselin değeri
# L = resimdeki maksimum ışık yoğunluğu
# görüntünün maksimum piksel değeri alınır
L = np.max(img)
# her pikselin makismum değerden çıkarılarak, negatif görüntü elde eder.
s = L - img
# yatay ve dikey birleştirme işlemleri gerçekleştirilir
hstacked = np.hstack((img, s))
vstacked = np.vstack((img, s))
plt.imshow(hstacked, cmap="gray")
plt.show()
