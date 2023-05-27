import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, filters, color, transform,util, segmentation, exposure,feature
from skimage.morphology import convex_hull_image
from scipy import ndimage as ndi
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from skimage.feature import local_binary_pattern


root = Tk()
root.withdraw()
root.directory = filedialog.askdirectory()
print (root.directory)
root.destroy()


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    if zoom_factor < 1:
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndi.zoom(img, zoom_tuple, **kwargs)
    elif zoom_factor > 1:
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndi.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    else:
        out = img
    return  exposure.rescale_intensity(out, in_range=(0, 1))


root.mainloop()

import os
list_gambar = ""

path = root.directory

l = os.listdir(path)
print("path"+path)

with open("ListGambar.txt", "w", encoding="utf-8") as file:
    for eachfile in l:
        if eachfile.endswith(('.png','.jpg','.jpeg')) == False:
            continue
        list_gambar += eachfile + "\n"
    file.write(list_gambar)

gambar = []
with open('ListGambar.txt', newline='') as infile:
    for line in infile:
        gambar.extend(line.strip().split(','))

alamatinput = root.directory + '\\'
alamato = 'Hasil Preproses - Kulit\\'

if not os.path.exists(alamato):
    os.makedirs(alamato)

r = 1
p = 8 * r

import cv2
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

urutan=-1
for n in gambar:
    urutan=urutan+1
    namakd = gambar[urutan].split(" ")
    print(namakd)
    namakd2 = (gambar[urutan].split("."))[0].split("_")
    print(namakd2)
    if not os.path.exists(alamato+namakd2[0]+'\\'):
        os.makedirs(alamato+namakd2[0]+'\\') #9 Kelas
    #if not os.path.exists(alamato+namakd2[0]+ "_" + namakd2[1]+'\\'):
    #   os.makedirs(alamato+namakd2[0]+ "_" + namakd2[1] + '\\') #45 Kelas - Binatang
    img = io.imread(alamatinput + gambar[urutan], as_gray=False)
    img = transform.resize(img, (100, 100), anti_aliasing=True)
    img = exposure.rescale_intensity(img, in_range=(0, 1))

    #=======================SETTINGS=====================================

    #img = filters.gaussian(img,0.35) #Gaussian

    #img = cv2.filter2D(src=img, ddepth=-1, kernel=sharpening_kernel) #Sharpening Using Custom Kernel (SUCK)

    # img = exposure.equalize_adapthist(util.img_as_ubyte(img), clip_limit=0.03) #CLAHE (Equalizer)

    #img = util.img_as_ubyte(img) #OPSIONAL (JIKA PERLU)

    #laplacian_without_gaussian = cv2.Laplacian(img, cv2.CV_64F) #Laplacian (Pt.1)
    #elev = np.uint8(np.abs(laplacian_without_gaussian)) #Laplacian (Pt.2)

    #elev = filters.difference_of_gaussians(img, 2, 10) #Difference Of Gaussians

    #B, G, R = cv2.split(img) #Canny Split (Pt.1)
    #B_cny = cv2.Canny(B, 0, 200,L2gradient=True) #Canny Split (Pt.2)
    #G_cny = cv2.Canny(G, 0, 200,L2gradient=True) #Canny Split (Pt.3)
    #R_cny = cv2.Canny(R, 0, 200,L2gradient=True) #Canny Split (Pt.4)
    #elev = cv2.merge([B_cny, G_cny, R_cny]) #Canny Split (Pt.5)

    #elev = filters.scharr(filters.prewitt(filters.gaussian(img,1))) #Gaussian + Prewitt + Scharr

    #elev = filters.prewitt(filters.sobel(img)) #Sobel + Prewitt

    #elev = filters.sobel(filters.prewitt(img)) #Prewitt + Sobel

    #elev, filt = filters.gabor(img, frequency=0.6) #Gabor (TESTING)

    #elev = local_binary_pattern(img, p, r, "uniform") #Local Binary Pattern

    #elev = filters.prewitt(img) #Prewitt

    #elev = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) #SOBEL

    #elev = util.img_as_ubyte(elev) #OPSIONAL (JIKA PERLU)

    #elev = color.rgb2gray(elev) #OPSIONAL (JIKA PERLU)

    #=======================SETTINGS=====================================


    elev = img
    print(elev)
    print(elev.shape)
    print(elev.size)
    print(elev.dtype)


    # imglabe= elev
    imglabe = img
    print(imglabe.dtype)
    print(imglabe.shape)
    print(imglabe.size)
    print(f'Shape: {imglabe.shape}')
    print("Values min/max", imglabe.min(),imglabe.max())

    #==========
    sca = [125, 129, 133, 137]
    rot = [75, 105, 175, 195]
    print(rot[1])

    for i in range(len(rot)):  # Rotasi
        print('Rotasi ' + str(rot[i]))
        rotasi = transform.rotate(imglabe, rot[i])
        x = 5
        y = 0
        rotasi = util.img_as_float(rotasi)
        for z in range(len(sca)):  # Scale
            stat = 100
            if sca[z] < 100:
                print(sca[z] / 100)
                imgzoom = clipped_zoom(rotasi, sca[z] / 100)
                stat = sca[z]  # Hanya status
            elif sca[z] >= 100:
                print(z / 100)
                imgzoom = clipped_zoom(rotasi, sca[z] / 100)
                stat = sca[z]  # Hanya status
            print(f'Shape: {imgzoom.shape}')
            print("a: ", imgzoom.min(), "b: ", imgzoom.max())
            x = 0


            im = np.array(imgzoom)

            # Work out where top left corner is
            yim = int((200 - 90) / 2)
            xim = int((200 - 90) / 2)

            # Crop, convert back from numpy to PIL Image and and save util.img_as_ubyte(cropninety)
            cropninety = im[xim:xim + 200, yim:yim + 200]

            io.imsave((alamato + namakd2[0]+ '\\' + (gambar[urutan].split("."))[0] + '_Scale_' + str(stat) + '%_Rotasi_' + str(rot[i]) + '.jpg'), util.img_as_ubyte(im))

