from PIL import Image
import numpy as np
import os

def clean_data(chemin_dossier_entree,chemin_dossier_sortie):
    for sous_dossier in os.listdir(chemin_dossier_entree):
        num = 0
        chemin_sous_dossier = os.path.join(chemin_dossier_entree, sous_dossier)
        chemin_sous_dossier_sortie = os.path.join(chemin_dossier_sortie, sous_dossier)
        
        if not os.path.exists(chemin_sous_dossier_sortie):
            os.makedirs(chemin_sous_dossier_sortie)
        for fichier in os.listdir(chemin_sous_dossier):
            chemin_fichier = os.path.join(chemin_sous_dossier, fichier)

            chemin_fichier_sortie = os.path.join(chemin_sous_dossier_sortie, sous_dossier + str(num)+'.jpg')
            
            image = Image.open(chemin_fichier)
            
            #dim du crop
            width, height = image.size
            if width > height:
                left = (width - height) // 2
                top = 0
                right = left + height
                bottom = height
            else:
                left = 0
                top = (height - width) // 2
                right = width
                bottom = top + width


            image = image.crop((left, top, right, bottom))
            image = image.resize((256,256))
            image = image.convert('L')

            image = normalize_grayscale_image(image)

            image.save(chemin_fichier_sortie, 'JPEG')


            print(chemin_fichier_sortie)
            num+=1


def normalize_grayscale_image(image):

    # Convertir l'image en tableau NumPy
    img_array = np.array(image, dtype=np.float32)

    # Normaliser les valeurs de pixel
    normalized_img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255.0

    # Convertir les valeurs normalisées en entiers 8 bits
    normalized_img_array = np.clip(normalized_img_array, 0, 255).astype(np.uint8)

    # Créer une nouvelle image à partir du tableau normalisé
    image = Image.fromarray(normalized_img_array)

    return image









