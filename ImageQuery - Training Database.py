import os
import h5py
import numpy as np
import argparse

import numpy as np
from numpy import linalg as LA
from skimage import io


from keras.preprocessing import image
from tensorflow.keras.models import Model

list_gambar = ""

#============SETTINGS=================


preprocess = 'Equalized'
cnn = 'MobileNetV2'


#============END=================


path = 'Hasil Preproses - Kulit - '+str(preprocess)+' - Test'


l = os.listdir(path)


with open("ListGambar.txt", "w", encoding="utf-8") as file:
    for eachfile in l:
        if eachfile.endswith(('.png', '.jpg', '.jpeg')) == False:
            continue
        list_gambar += eachfile + "\n"
    file.write(list_gambar)

gambar = []
with open('ListGambar.txt', newline='') as infile:
    for line in infile:
        gambar.extend(line.strip().split(','))

def get_imlist(path):

    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


urutan = -1



class KerasApp:
    def __init__(self, cnn):
        if cnn=="ResNet50V2":
            from keras.applications.resnet_v2 import ResNet50V2


            base_model = ResNet50V2(weights='imagenet')
        elif cnn=="MobileNetV2":
            from keras.applications.mobilenet_v2 import MobileNetV2


            base_model = MobileNetV2(weights='imagenet')

        elif cnn=="EfficientNetB7":
            from tensorflow.keras.applications.efficientnet import EfficientNetB7


            base_model = EfficientNetB7(weights='imagenet')

        elif cnn=="InceptionV3":
            from keras.applications.inception_v3 import InceptionV3


            base_model = InceptionV3(weights='imagenet')




        base_model.summary()
        self.model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    def extract_feat(self, img, cnn):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)
        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """

        if cnn == "ResNet50V2":
            from keras.applications.resnet_v2 import preprocess_input

            img = image.load_img(img, target_size=(224,224), color_mode='rgb')

        elif cnn == "MobileNetV2":
            from keras.applications.mobilenet_v2 import preprocess_input

            img = image.load_img(img, target_size=(224,224), color_mode='rgb')

        elif cnn == "EfficientNetB7":
            from tensorflow.keras.applications.efficientnet import preprocess_input

            img = image.load_img(img, target_size=(600, 600), color_mode='rgb')

        elif cnn == "InceptionV3":
            from keras.applications.inception_v3 import preprocess_input

            img = image.load_img(img, target_size=(299, 299), color_mode='rgb')


        #img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        norm_feat = feature / np.linalg.norm(feature)  # Normalize
        return norm_feat


'''
 Extract features and index the images
'''



if __name__ == "__main__":

    #db = args["database"]
    img_list = get_imlist(path)
    #print(os.listdir(path))
    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    model = KerasApp(cnn)
    #print(img_list)
    for i, img_path in enumerate(img_list):
        #print("ASDASD")

        #norm_feat = model.extract_feat(img_path)
        norm_feat = model.extract_feat(img_path, cnn)
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))

    feats = np.array(feats)
    print(feats.shape)
    # directory for storing extracted features
    #output = args["index"]

    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File("TemuKembaliKulit5"+str(preprocess)+str(cnn)+".h5", 'w')
    h5f.create_dataset('dataset_1', data=feats)
    ## h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()