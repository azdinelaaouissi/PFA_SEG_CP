import os
import streamlit as st
import streamlit_image_comparison as sic
from PIL import Image
import base64

# Configuration de la page
st.set_page_config(layout='wide')

# Titre principal
st.title("Projet de Détection du Cancer du Poumon")

# Ajout de CSS personnalisé pour ajuster l'affichage des images
st.markdown(
    """
    <style>
        .stImageComparison {
            display: flex;
            justify-content: center;
        }
        .stImageComparison img {
            margin: 0 10px;
            width: 300px;
        }
        .text-container {
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: left;
            padding: 10px;
        }
        .toto {
            padding-left: 20px;
            text-align: center;
            margin-top: 20%;
        }
        .video-container {
            display: flex;
            justify-content: center;
        }
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            text-align: center;
        }
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin: 20px auto;
        }
        .image-container div {
            text-align: center;
        }
        .image-container img {
            max-width: 300px;
            height: auto;
        }
        .image-container p {
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def get_base64_image(image_path):
    """Convertir une image en une chaîne base64 pour l'affichage HTML."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_image(image_path, target_width, target_height):
    """Redimensionner l'image à une largeur et une hauteur fixes."""
    with Image.open(image_path) as img:
        img.thumbnail((target_width, target_height), Image.LANCZOS)
        img_resized = img.resize((target_width, target_height), Image.LANCZOS)
    return img_resized

# Comparaison de quatre paires d'images
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# Chemins des images pour la comparaison
image1_normal_path = "image1_normal.png"
image1_with_mask_path = "image1_with_mask.png"
image2_normal_path = "image_normale2.png"
image2_with_mask_path = "image_superposee2.png"
image3_normal_path = "image3_normal.png"
image3_with_mask_path = "image3_with_mask.png"
image4_normal_path = "image_normale.png"
image4_with_mask_path = "image_superposee.png"

# Comparaison de l'image 1
with col1:
    if os.path.exists(image1_normal_path) and os.path.exists(image1_with_mask_path):
        sic.image_comparison(
            img1=image1_normal_path,
            img2=image1_with_mask_path,
            label1="Image 1 Normale",
            label2="Image 1 Superposée avec Masque",
            width=250
        )
    else:
        st.error("Les images pour la comparaison 1 ne sont pas trouvées.")

# Comparaison de l'image 2
with col2:
    if os.path.exists(image2_normal_path) and os.path.exists(image2_with_mask_path):
        sic.image_comparison(
            img1=image2_normal_path,
            img2=image2_with_mask_path,
            label1="Image 2 Normale",
            label2="Image 2 Superposée avec Masque",
            width=250
        )
    else:
        st.error("Les images pour la comparaison 2 ne sont pas trouvées.")

# Comparaison de l'image 3
with col3:
    if os.path.exists(image3_normal_path) and os.path.exists(image3_with_mask_path):
        sic.image_comparison(
            img1=image3_normal_path,
            img2=image3_with_mask_path,
            label1="Image 3 Normale",
            label2="Image 3 Superposée avec Masque",
            width=250
        )
    else:
        st.error("Les images pour la comparaison 3 ne sont pas trouvées.")

# Comparaison de l'image 4
with col4:
    if os.path.exists(image4_normal_path) and os.path.exists(image4_with_mask_path):
        sic.image_comparison(
            img1=image4_normal_path,
            img2=image4_with_mask_path,
            label1="Image 4 Normale",
            label2="Image 4 Superposée avec Masque",
            width=250
        )
    else:
        st.error("Les images pour la comparaison 4 ne sont pas trouvées.")

# Section pour afficher l'image NIfTI et son masque
st.header("Image NIfTI et Masque")

nifti_image_path = "nifti_image.png"
nifti_mask_path = "nifti_mask.png"

# Vérifiez si les fichiers existent avant d'essayer de les afficher
if os.path.exists(nifti_image_path) and os.path.exists(nifti_mask_path):
    st.markdown(
        """
        <div class="image-container">
            <div>
                <img src="data:image/png;base64,{}" alt="Image NIfTI" />
                <p>Image NIfTI</p>
            </div>
            <div>
                <img src="data:image/png;base64,{}" alt="Masque NIfTI" />
                <p>Masque NIfTI</p>
            </div>
        </div>
        """.format(
            get_base64_image(nifti_image_path),
            get_base64_image(nifti_mask_path)
        ),
        unsafe_allow_html=True
    )
else:
    st.error("L'une ou plusieurs des images NIfTI ne sont pas trouvées.")

# Section pour afficher la vidéo avec le texte explicatif
st.header("Animation NIfTI")

video_file = "téléchargement.mp4"
video_file2 = "ct_scan_animation_green_contours (1).mp4"

# Conteneur pour les deux vidéos et le texte
col6, col7, col8 = st.columns([2, 1, 1])  # Ajustez les tailles des colonnes

# Affichage de la première vidéo à gauche
with col6:
    st.markdown(
        """
        <div class='text-container toto'>
            <p>
            Cette animation montre l'évolution des coupes d'images médicales en NIfTI avec les prédictions du masque superposé. La répétition automatique permet de visualiser en continu les résultats.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Texte explicatif centré
with col7:
    if os.path.exists(video_file2):
        st.video(video_file2, loop=True)
    else:
        st.error(f"Le fichier vidéo '{video_file2}' n'est pas trouvé.")

# Affichage de la deuxième vidéo à droite
with col8:
    if os.path.exists(video_file):
        st.video(video_file, loop=True)
    else:
        st.error(f"Le fichier vidéo '{video_file}' n'est pas trouvé.")
