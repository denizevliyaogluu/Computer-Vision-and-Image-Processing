#     #8 tane piksel vardı hepsine 10 değerine atadı
#     a=np.array([1,2,3,4,5,6,7,8])
#     b=np.full_like(a,10)
#     print(b)
#     #değiştirilecek pikselleri, 5 ve 5'ten buyuk olanlara false değeri kalana true değeri
#     pk = (a>5)
#     print(pk)

#     #true değerlerini c pikseli olarak ver , 5'ten buyuk olanlar gelir
#     c=a[pk]
#     print(c)

#     #true olanlara 255 değerini verdi
#     # a[pk] = 255
#     # print(a)

#     #iki değerim var. 4 ve 5'e true diğerlerine false verdi
#     pk = np.logical_and(a>3,a<6)
#     print(pk)
#     #4 ve 5'in oldğu yere 255 değerini verdi
#     a[pk] = 255
#     print(a)

#A ve B aralığının dışında kalanlar alt değer, içinde kalanlar ust deger
import cv2
import numpy as np
import matplotlib.pyplot as plt

def binary_slicing(img,A,B,alt,ust):
    img_out = np.full_like(img, alt)
    #resmin piksellerinin A'dan büyük olan kısmını al ve img'in B'den küçük olan kısımlarını al. Yani A ve B arasındaki pikselleri true false olarak aldık
    pk = np.logical_and(img >= A, img<=B)
    img_out[pk] = ust
    return img_out

def linear_slicing(img,A,B,ust):
    img_out=img.copy()
    pk=np.logical_and(img >= A, img <=B)
    img_out[pk] = ust
    return img_out

def linear_slicing_reverse(img,A,B,ust):
    img_out=img.copy()
    pk_kucuk = np.logical_and(img>=0, img<=A)
    pk_buyuk = np.logical_and(img>=B, img<=255)
    pk = np.logical_or(pk_kucuk, pk_buyuk)
    img_out[pk]=ust
    return img_out

img_path="./img/aortic_angiohram.tif"
img=cv2.imread(img_path, 0)
#A ve B arasındaki değerlere 255, dışındaki değerlere 10 değerini ver
A=150
B=200
ust=255
alt=10

bs_img = binary_slicing(img,A,B,alt,ust)
ls_img = linear_slicing(img,A,B,ust)
lsr_img = linear_slicing_reverse(img,A,B,ust)

hstacked1 = np.hastack((img,bs,img))
hstacked2 = np.hastcak((ls_img, lsr_img))
vstacked = np.vstack((hstacked1, hstacked2))

plt.imshow(vstacked, cmap="gray")
plt.show
