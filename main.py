import albumentations as A
import cv2
import numpy as np
import os

#Correção do BGR para RGB
def load_img(path):
    image = cv2.imread(path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

#Carregar imagem
input_path = r''

output_path = r''


#Aumentation pipeline com Albumetations
#Tem possibilidade de 100% (p=1) de aplicar blur. O 60,70 é o range possível de desfoque
p = 0.2
transform_blur = A.Compose([
    A.Resize(width=640, height=640),
    A.RandomResizedCrop(width=256, height=256, scale=(0.8, 1.0), p=p),
    A.Rotate(limit=45, p=p),
    A.HorizontalFlip(p=p),
    A.Transpose(p=p),
    A.ElasticTransform(alpha=1.5, sigma=0.1, p=p),
    A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=p),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=p),
    A.RandomBrightnessContrast(p=p),
    A.HueSaturationValue(p=p),
    A.RGBShift(p=p),
    A.RandomGamma(p=p),
    A.Blur(blur_limit=(3, 7), p=p),
    A.GaussNoise(var_limit=(10.0, 80.0), p=p),
    A.RandomCrop(height=200, width=200, p=p),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=p),
    A.RandomGridShuffle(grid=(3, 3), p=p),
    A.RandomShadow(p=p),
    A.ChannelDropout(channel_drop_range=(1, 2), p=0),
    A.ChannelShuffle(p=0)
    ])

transform_names = ["Resize","RandomResizedCrop","Rotate","HorizontalFlip","Transpose","ElasticTransform","OpticalDistortion","GridDistortion","RandomBrightnessContrast","HueSaturationValue","RGBShift","RandomGamma","Blur","GaussNoise","RandomCrop","GridDistortion","RandomGridShuffle","RandomShadow","ChannelDropout","ChannelShuffle"]

# Defina o número de imagens aleatórias que você deseja gerar
num_generated_images = 200

# Aplique as transformações e salve as imagens geradas
for arquivo in os.listdir(input_path):
    print(arquivo)
    input_img = load_img(os.path.join(input_path,arquivo))
    #input_img = input_img[...,:3][...,::-1]
    for i in range(0,num_generated_images):
        transformed = transform_blur(image=input_img)
        transformed_image = transformed['image']
        
        arquivo_s_ext = os.path.splitext(arquivo)[0] #Pega o nome do arquvi sem a extensão
        named = os.path.join(output_path,f'{arquivo_s_ext}_generated_image_{i+1}.jpg')
        cv2.imwrite(named, transformed_image)