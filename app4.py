# import streamlit as st
# import numpy as np
# import cv2
# import torch
# import nibabel as nib
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from TumorSegmentation import *  # Assurez-vous que ce module est bien implémenté
# from io import BytesIO
# import  io
# from skimage import measure
#
# # Charger le modèle (assurez-vous que le chemin vers le modèle est correct)
# path_model = "epoch=27-step=48580.ckpt"
# model = TumorSegmentation.load_from_checkpoint(path_model)
# model.eval()
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# model.to(device)
#
# # Fonction pour prédire le masque
# def predict_mask(image_path):
#     image = np.load(image_path)
#     image = cv2.resize(image, (256, 256))  # Redimensionner si nécessaire
#     image_tensor = torch.tensor(image).float().to(device).unsqueeze(0).unsqueeze(0)
#
#     with torch.no_grad():
#         pred_mask = torch.sigmoid(model(image_tensor))
#         pred_mask = pred_mask[0][0].cpu().numpy()
#         pred_mask_binary = pred_mask > 0.5  # Binariser la prédiction
#
#     return image, pred_mask_binary
# # Fonction pour lire et prédire à partir d'un fichier .nii.gz
# def read_nifti(file):
#     file.seek(0)  # Réinitialiser le pointeur de fichier pour la lecture
#     nifti_data = nib.load(io.BytesIO(file.read())).get_fdata() / 3071
#     return nifti_data
# # Streamlit application
# st.title("Segmentation de Cancer du Poumon")
#
#
# # Section pour le fichier NumPy (.npy)
# st.markdown("### Charger un fichier NumPy (.npy)")
# uploaded_npy = st.file_uploader("Choisissez un fichier NumPy", type=['npy'])
#
# if uploaded_npy is not None:
#     try:
#         # Prédire et générer les images
#         image, pred_mask_binary = predict_mask(uploaded_npy)
#
#         # Affichage des images en une seule ligne avec trois colonnes
#         col1, col2, col3 = st.columns(3)
#
#         # Image Normale avec cmap="bone"
#         with col1:
#             fig, ax = plt.subplots()
#             ax.imshow(image, cmap="bone")
#             ax.axis('off')  # Supprimer les axes pour une présentation propre
#             st.pyplot(fig, bbox_inches='tight', pad_inches=0)
#             st.markdown("<p style='text-align: center;'>Image Normale</p>", unsafe_allow_html=True)  # Ajouter le titre centré sous l'image
#
#         # Masque Prédit avec cmap="gray"
#         with col2:
#             fig, ax = plt.subplots()
#             ax.imshow(pred_mask_binary, cmap="gray")
#             ax.axis('off')  # Supprimer les axes pour une présentation propre
#             st.pyplot(fig, bbox_inches='tight', pad_inches=0)
#             st.markdown("<p style='text-align: center;'>Le Masque</p>", unsafe_allow_html=True)  # Ajouter le titre centré sous l'image
#
#         # Superposition de l'image normale et du masque prédit
#         with col3:
#             contours = measure.find_contours(pred_mask_binary, 0.5)
#
#             fig, ax = plt.subplots()
#             ax.imshow(image, cmap="bone")
#             ax.imshow(np.ma.masked_where(pred_mask_binary == 0, pred_mask_binary), alpha=0.5, cmap="autumn")
#             for contour in contours:
#                 ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red', alpha=0.8)  # Ajouter les contours rouges
#             ax.axis('off')  # Supprimer les axes pour une présentation propre
#             st.pyplot(fig, bbox_inches='tight', pad_inches=0)
#             st.markdown("<p style='text-align: center;'>Image Superposée</p>", unsafe_allow_html=True)  # Ajouter le titre centré sous l'image
#
#     except Exception as e:
#         st.error(f"Erreur lors de la prédiction : {str(e)}")
#
# st.markdown("---")
#
# # Section pour le fichier NIfTI (.nii.gz)
# st.markdown("### Charger un fichier NIfTI (.nii.gz)")
#
# # Charger le fichier NIfTI
# uploaded_nifti = st.file_uploader("Choisissez un fichier NIfTI (.nii.gz)", type=['nii', 'nii.gz'])
#
# if uploaded_nifti is not None:
#     try:
#         # Lire les données du fichier NIfTI
#         ct_scan = read_nifti(uploaded_nifti)
#         ct_scan = ct_scan[:, :, 30:]  # Modifier si nécessaire
#
#         segmentation, scan = [], []
#
#         for i in range(ct_scan.shape[-1]):
#             slice = ct_scan[:, :, i]
#             slice = cv2.resize(slice, (256, 256))
#             slice_tensor = torch.tensor(slice).unsqueeze(0).unsqueeze(0).float().to(device)
#
#             with torch.no_grad():
#                 pred = model(slice_tensor)[0][0].cpu()
#             pred_binary = pred > 0.5
#             segmentation.append(pred_binary)
#             scan.append(slice)
#
#         # Créer et afficher la vidéo
#         video_bytes = create_video(scan, segmentation, device)
#         st.video(video_bytes, format='video/mp4', start_time=0)
#
#     except Exception as e:
#         st.error(f"Erreur lors de la prédiction : {str(e)}")
#
# # Footer
# st.markdown("<footer style='text-align: center; margin-top: 50px;'>"
#             "Projet de segmentation du cancer du poumon - Développé avec Streamlit"
#             "</footer>", unsafe_allow_html=True)
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from celluloid import Camera
import nibabel as nib
import cv2
import tempfile
import base64
from TumorSegmentation import TumorSegmentation
from skimage import measure  # Import for contour detection

# Charger le modèle

# def load_model():
#     path_model = "epoch=27-step=48580.ckpt"  # Chemin vers le modèle entraîné
#     model = TumorSegmentation.load_from_checkpoint(path_model)
#     model.eval()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     return model, device
#
# # Fonction pour prédire le masque
# def predict_mask(image_path):
#     image = np.load(image_path)
#     image = cv2.resize(image, (256, 256))  # Redimensionner si nécessaire
#     image_tensor = torch.tensor(image).float().to(device).unsqueeze(0).unsqueeze(0)
#
#     with torch.no_grad():
#         pred_mask = torch.sigmoid(model(image_tensor))
#         pred_mask = pred_mask[0][0].cpu().numpy()
#         pred_mask_binary = pred_mask > 0.5  # Binariser la prédiction
#
#     return image, pred_mask_binary
#
# # Fonction pour lire et prédire à partir d'un fichier .nii.gz
# def read_nifti(file):
#     file.seek(0)  # Réinitialiser le pointeur de fichier pour la lecture
#     nifti_data = nib.load(io.BytesIO(file.read())).get_fdata() / 3071
#     return nifti_data
#
# # Chargement du modèle
# model, device = load_model()
#
# # Interface utilisateur de Streamlit
# st.title("Segmentation de Cancer du Poumon")
#
# # Partie pour charger le fichier NumPy (.npy)
# st.markdown("### Charger un fichier NumPy (.npy)")
# uploaded_npy = st.file_uploader("Choisissez un fichier NumPy", type=['npy'])
#
# if uploaded_npy is not None:
#     try:
#         # Prédire et générer les images
#         image, pred_mask_binary = predict_mask(uploaded_npy)
#
#         # Affichage des images en une seule ligne avec trois colonnes
#         col1, col2, col3 = st.columns(3)
#
#         # Image Normale avec cmap="bone"
#         with col1:
#             fig, ax = plt.subplots()
#             ax.imshow(image, cmap="bone")
#             ax.axis('off')
#             st.pyplot(fig, bbox_inches='tight', pad_inches=0)
#             st.markdown("<p style='text-align: center;'>Image Normale</p>", unsafe_allow_html=True)
#
#         # Masque Prédit avec cmap="gray"
#         with col2:
#             fig, ax = plt.subplots()
#             ax.imshow(pred_mask_binary, cmap="gray")
#             ax.axis('off')
#             st.pyplot(fig, bbox_inches='tight', pad_inches=0)
#             st.markdown("<p style='text-align: center;'>Le Masque</p>", unsafe_allow_html=True)
#
#         # Superposition de l'image normale et du masque prédit
#         with col3:
#             contours = measure.find_contours(pred_mask_binary, 0.5)
#             fig, ax = plt.subplots()
#             ax.imshow(image, cmap="bone")
#             ax.imshow(np.ma.masked_where(pred_mask_binary == 0, pred_mask_binary), alpha=0.5, cmap="autumn")
#             for contour in contours:
#                 ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red', alpha=0.8)
#             ax.axis('off')
#             st.pyplot(fig, bbox_inches='tight', pad_inches=0)
#             st.markdown("<p style='text-align: center;'>Image Superposée</p>", unsafe_allow_html=True)
#     except Exception as e:
#         st.error(f"Erreur lors de la prédiction : {str(e)}")
#
# st.markdown("---")
#
# # Partie pour charger le fichier NIfTI (.nii.gz)
# st.markdown("### Charger un fichier NIfTI (.nii.gz)")
# uploaded_file = st.file_uploader("Choisissez un fichier NIfTI (.nii.gz)", type=['nii', 'nii.gz'])
#
#
# if uploaded_file is not None:
#     # Lire le fichier NIfTI
#     try:
#         # Créer un fichier temporaire avec l'extension correcte
#         suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_path = temp_file.name
#
#         # Charger le fichier avec nibabel
#         ct_scan = nib.load(temp_path).get_fdata() / 3071  # Normalisation des valeurs du scan
#         ct_scan = ct_scan[:, :, 30:]  # Optionnel : sélectionner une tranche spécifique
#
#         # Variables pour stocker les prédictions et les images CT
#         segmentation, scan = [], []
#
#         # Boucle pour redimensionner et faire les prédictions sur chaque tranche
#         for i in range(ct_scan.shape[-1]):
#             slice_ = ct_scan[:, :, i]  # Extraire une tranche
#             slice_resized = cv2.resize(slice_, (256, 256))  # Redimensionner
#             scan.append(slice_resized)  # Ajouter l'image CT à la liste
#
#             slice_tensor = torch.tensor(slice_resized).unsqueeze(0).unsqueeze(0).float().to(device)  # Préparer pour le modèle
#
#             # Faire la prédiction avec le modèle
#             with torch.no_grad():
#                 pred = model(slice_tensor)[0][0].cpu()
#
#             pred_binary = pred > 0.5  # Binarisation du masque (seuil à 0.5)
#             segmentation.append(pred_binary.numpy())  # Ajouter le masque à la liste
#
#         # Création de l'animation avec les images et les masques
#         fig = plt.figure(figsize=(4, 4))  # Taille de la figure
#         camera = Camera(fig)
#
#         # Boucle pour afficher chaque image et le masque correspondant
#         for i in range(0, len(scan), 2):  # Afficher toutes les 2 tranches pour accélérer l'animation
#             plt.imshow(scan[i], cmap="bone")  # Image CT-scan
#             mask = np.ma.masked_where(segmentation[i] == 0, segmentation[i])  # Masquer les zones sans prédiction
#             plt.imshow(mask, alpha=0.5, cmap="autumn")  # Superposer le masque
#
#             # Trouver les contours dans le masque binaire
#             contours = measure.find_contours(segmentation[i], level=0.5)
#             # Ajouter les contours en rouge
#             for contour in contours:
#                 plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # Tracer les contours en rouge
#
#             plt.axis("off")  # Supprimer les axes
#             plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ajuster les marges pour supprimer les bordures blanches
#             plt.tight_layout(pad=0)  # Ajuster les espaces pour supprimer les bordures blanches
#             camera.snap()
#
#         # Générer l'animation
#         animation = camera.animate(interval=50)  # Intervalle de 50ms entre les frames
#         plt.close()
#
#         # Sauvegarder la vidéo dans un fichier temporaire
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
#             animation.save(video_file.name)  # Enregistrer la vidéo
#             video_path = video_file.name
#
#         # Lire le contenu du fichier pour l'encodage en base64
#         with open(video_path, "rb") as video_file:
#             video_data = video_file.read()
#             video_base64 = base64.b64encode(video_data).decode("utf-8")
#
#         # Créer un lecteur vidéo avec la taille personnalisée via HTML
#         # Créer un lecteur vidéo centré avec la taille personnalisée via HTML et du CSS
#         video_html = f"""
#             <div style="display: flex; justify-content: center; align-items: center;">
#                 <video width="350" controls loop>
#                     <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
#                     Your browser does not support the video tag.
#                 </video>
#             </div>
#         """
#         st.markdown(video_html, unsafe_allow_html=True)
#
#
#     except Exception as e:
#         st.error(f"Erreur lors du chargement ou du traitement du fichier : {e}")
#
# # Footer
# st.markdown("<footer style='text-align: center; margin-top: 50px;'>"
#             "Projet de segmentation du cancer du poumon - Développé avec Streamlit"
#             "</footer>", unsafe_allow_html=True)
#
#


#
# ######################################################################"
# import streamlit as st
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from celluloid import Camera
# import nibabel as nib
# import cv2
# import tempfile
# import base64
# from TumorSegmentation import TumorSegmentation
# from skimage import measure  # Import for contour detection
#
# # Charger le modèle
# @st.cache(allow_output_mutation=True)
# def load_model():
#     path_model = "epoch=27-step=48580.ckpt"  # Chemin vers le modèle entraîné
#     model = TumorSegmentation.load_from_checkpoint(path_model)
#     model.eval()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     return model, device
#
# model, device = load_model()
#
# # Interface utilisateur de Streamlit
# st.title("Segmentation de Scan Pulmonaire")
#
# # Charger le fichier .nii
# uploaded_file = st.file_uploader("Télécharger un fichier de scan (.nii ou .nii.gz)", type=["nii", "nii.gz"])
#
# if uploaded_file is not None:
#     # Lire le fichier NIfTI
#     try:
#         # Créer un fichier temporaire avec l'extension correcte
#         suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_path = temp_file.name
#
#         # Charger le fichier avec nibabel
#         ct_scan = nib.load(temp_path).get_fdata() / 3071  # Normalisation des valeurs du scan
#         ct_scan = ct_scan[:, :, 30:]  # Optionnel : sélectionner une tranche spécifique
#
#         # Variables pour stocker les prédictions et les images CT
#         segmentation, scan = [], []
#
#         # Boucle pour redimensionner et faire les prédictions sur chaque tranche
#         for i in range(ct_scan.shape[-1]):
#             slice_ = ct_scan[:, :, i]  # Extraire une tranche
#             slice_resized = cv2.resize(slice_, (256, 256))  # Redimensionner
#             scan.append(slice_resized)  # Ajouter l'image CT à la liste
#
#             slice_tensor = torch.tensor(slice_resized).unsqueeze(0).unsqueeze(0).float().to(device)  # Préparer pour le modèle
#
#             # Faire la prédiction avec le modèle
#             with torch.no_grad():
#                 pred = model(slice_tensor)[0][0].cpu()
#
#             pred_binary = pred > 0.5  # Binarisation du masque (seuil à 0.5)
#             segmentation.append(pred_binary.numpy())  # Ajouter le masque à la liste
#
#         # Création de l'animation avec les images et les masques
#         fig = plt.figure(figsize=(4, 4))  # Taille de la figure
#         camera = Camera(fig)
#
#         # Boucle pour afficher chaque image et le masque correspondant
#         for i in range(0, len(scan), 2):  # Afficher toutes les 2 tranches pour accélérer l'animation
#             plt.imshow(scan[i], cmap="bone")  # Image CT-scan
#             mask = np.ma.masked_where(segmentation[i] == 0, segmentation[i])  # Masquer les zones sans prédiction
#             plt.imshow(mask, alpha=0.5, cmap="autumn")  # Superposer le masque
#
#             # Trouver les contours dans le masque binaire
#             contours = measure.find_contours(segmentation[i], level=0.5)
#             # Ajouter les contours en rouge
#             for contour in contours:
#                 plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)  # Tracer les contours en rouge
#
#             plt.axis("off")  # Supprimer les axes
#             plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ajuster les marges pour supprimer les bordures blanches
#             plt.tight_layout(pad=0)  # Ajuster les espaces pour supprimer les bordures blanches
#             camera.snap()
#
#         # Générer l'animation
#         animation = camera.animate(interval=50)  # Intervalle de 50ms entre les frames
#         plt.close()
#
#         # Sauvegarder la vidéo dans un fichier temporaire
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
#             animation.save(video_file.name)  # Enregistrer la vidéo
#             video_path = video_file.name
#
#         # Lire le contenu du fichier pour l'encodage en base64
#         with open(video_path, "rb") as video_file:
#             video_data = video_file.read()
#             video_base64 = base64.b64encode(video_data).decode("utf-8")
#
#         # Créer un lecteur vidéo avec la taille personnalisée via HTML
#         video_html = f"""
#             <video width="300" controls>
#                 <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
#                 Your browser does not support the video tag.
#             </video>
#         """
#         st.markdown(video_html, unsafe_allow_html=True)
#
#         # Permettre de télécharger la vidéo
#         with open(video_path, "rb") as file:
#             st.download_button("Télécharger la vidéo", file, file_name="segmentation_video.mp4")
#
#     except Exception as e:
#         st.error(f"Erreur lors du chargement ou du traitement du fichier : {e}")

import streamlit as st

# Désactiver les avertissements de Streamlit pour le HTML non sécurisé

import streamlit as st

# Titre du projet
st.title("Résumé du Projet")

# Description du projet
st.write(
    "Le travail présenté dans ce rapport, réalisé au sein de la Faculté des Sciences Ben M'Sik, "
    "s'inscrit dans le cadre du projet de fin d'année pour l'obtention du diplôme de Master en Data Science et Big Data. "
    "L'objectif de ce projet est de développer une solution efficace pour la segmentation d'images médicales, "
    "avec une application spécifique au cancer du poumon et à la segmentation des images cardiaques."
)

st.write(
    "Afin de répondre à cet objectif, une architecture **U-Net** a été mise en place pour chaque cas étudié. "
    "La segmentation des images pulmonaires a été réalisée en utilisant PyTorch, tandis que les images cardiaques ont été "
    "traitées avec TensorFlow et Keras. Ce choix permet de comparer les performances des deux frameworks dans un contexte médical. "
    "Une attention particulière a été portée à l’évaluation des modèles, notamment en termes de précision et de temps d'exécution."
)

st.write(
    "Pour faciliter l'interprétation des résultats, une interface de visualisation interactive a été développée en utilisant Streamlit, "
    "permettant de visualiser les images segmentées et de mieux comprendre les performances des différents modèles."
)

st.write(
    "Ce projet démontre l’importance des techniques de deep learning dans l’amélioration du diagnostic médical et propose une solution complète, "
    "alliant performance, flexibilité et accessibilité à travers une interface interactive pour les utilisateurs."
)

st.write(
    "Ce projet se concentre sur **la segmentation d'images médicales** pour la détection d'anomalies associées à deux types de maladies : "
    "le cancer du poumon et le cancer du cœur. Deux ensembles de données distincts sont utilisés :"
)

# Listes des datasets
st.markdown("""
- [Dataset pour le cancer du poumon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) : constitué d'images CT provenant de la base de données NCI, permettant d'identifier les lésions pulmonaires.
- [Dataset pour les maladies cardiaques](https://www.kaggle.com/datasets/nikhilroxtomar/ct-heart-segmentation) : comprenant des images PNG représentant divers cas de maladies cardiaques.
""")

st.write(
    "Pour la segmentation, nous adoptons l'architecture **U-Net**, reconnue pour son efficacité dans l'identification des régions d'intérêt dans les images médicales. "
    "Deux modèles distincts sont développés :"
)

# Modèles de segmentation
st.markdown("""
- **U-Net avec PyTorch** : conçu pour traiter les images CT du cancer du poumon, utilisant des techniques de **data augmentation** et des méthodes d'optimisation adaptées.
- **U-Net avec TensorFlow/Keras** : ciblant la segmentation d'images liées aux maladies cardiaques, intégrant également des stratégies d'augmentation des données.
""")

st.write(
    "Les résultats de ce projet visent à fournir des outils d'assistance aux cliniciens pour améliorer le diagnostic et le suivi des patients atteints de cancer du poumon et de maladies cardiaques."
)

st.write(
    "**Code du projet disponible sur GitHub :** [GitHub Repository](https://github.com/azdinelaaouissi/seg_projet)"
)
