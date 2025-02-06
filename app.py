import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt

import streamlit_image_comparison as sic
from skimage import measure
import nibabel as nib
from nilearn import plotting
from matplotlib.animation import FuncAnimation
import io
import os
import  pandas  as pd
import torch
import matplotlib.pyplot as plt
from celluloid import Camera
import nibabel as nib
import cv2
import tempfile
import base64
from TumorSegmentation import TumorSegmentation
from skimage import measure  # Import for contour detection


# Configuration de la page
st.set_page_config(page_title="Application Médicale", layout="wide")

# Style CSS pour personnaliser les tailles et les icônes
st.markdown("""
    <style>
    .toto{
          margin-top:-30px;
    }
    .abstract-title {
        font-size: 30px !important;
        font-weight: bold;
        display: flex;
        align-items: center;
        margin-bottom: 0px;
    }
    .abstract-text {
        font-size: 18px;
        margin-bottom: 15px;
      
        list-style-type: none; /* Désactiver les puces de li */
        padding-left: 0; /* Enlever le padding par défaut de ul */
    }
    .abstract-text li {
        margin-bottom: 10px;
    }
    .abstract-text li:before {
        content: "\\2192"; /* Unicode de la flèche droite */
        margin-right: 8px;
        font-weight: bold;
        color: #007BFF;
    }
    .image-column {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    .text-column {
        padding: 20px;
    }
    .content-column {
        margin-top: 20px; /* Ajouter de l'espace vertical entre les colonnes */
    }
    </style>
    """, unsafe_allow_html=True)


# Fonction pour le menu principal
def display_main_menu():
    return option_menu(
        menu_title=None,  # Pas de titre pour le menu
        options=["Abstract", "Exemple", "Heart", "Lung Cancer"],  # Options du menu
        icons=['file-text', 'book', 'heart', 'lungs'],  # Icônes pour chaque bouton
        menu_icon="cast",  # Icône principale du menu (non utilisée ici)
        default_index=0,  # Option sélectionnée par défaut
        orientation="horizontal"  # Orientation horizontale
    )


# Menu principal horizontal
selected_main = display_main_menu()


# Fonction pour afficher le contenu principal
def display_content():
    if selected_main == "Abstract":
        st.markdown('<h1 class="abstract-title">📗 Abstract</h1>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])  # Ratio de largeur des colonnes (1:2)

        with col1:
            # Affichage de l'image avec une largeur de 400px
            image = Image.open('41467_2024_44824_Fig1_HTML.png')
            st.image(image, caption='Segmentation Médicale', width=380)
            image = Image.open('u-net-architecture.png')
            st.image(image, caption='u-net architecture', width=380)
            image = Image.open('1_6aYcfNbhjJH8sl1lN6YXDA.png')
            st.image(image, caption='Pytorch and  Tensorflow ', width=380)
        with col2:
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






    elif selected_main == "Exemple":
        st.title("🚨 Exemple")

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


    elif selected_main in ["Heart", "Lung Cancer"]:
        col1, col2 = st.columns([1, 3])  # Ajustez les ratios selon vos préférences

        with col1:
            # Sous-menu vertical dans la colonne 1
            selected_sub = option_menu(
                menu_title=None,  # Pas de titre pour le sous-menu
                options=["Visualisation et Métriques", "Tester ou Prédire"],  # Options du sous-menu
                icons=['bar-chart', 'robot'],  # Icônes pour les options du sous-menu
                menu_icon="heart" if selected_main == "Heart" else "lungs",  # Icône pour le sous-menu
                default_index=0,  # Option sélectionnée par défaut
                orientation="vertical"  # Orientation verticale du sous-menu
            )

        with col2:
            # Display content corresponding to the selected submenu
            if selected_sub == "Visualisation et Métriques":
                if selected_main == "Heart":
                    st.write("Affichage des métriques et des visualisations pour les maladies cardiaques.")
                elif selected_main == "Lung Cancer":
                    # Display titles and images with captions
                    st.subheader("🩺 Dice Score vs Threshold")
                    dice_img = Image.open('foo.png')
                    st.image(dice_img, caption='Dice Score vs Threshold')

                    st.subheader("📊 Precision-Recall Curve")
                    pr_img = Image.open('precision_recall_curve.png')
                    st.image(pr_img, caption='Precision-Recall Curve')

                    st.subheader("📈 ROC Curve")
                    roc_img = Image.open('roc_curve.png')
                    st.image(roc_img, caption='ROC Curve')

                    st.subheader("📉 Calibration Curve")
                    calib_img = Image.open('calibration_curve.png')
                    st.image(calib_img, caption='Calibration Curve')

                    metrics = {
                        "Metric": ["IoU", "Precision", "Recall", "F1 Score", "Accuracy", "Dice Score"],
                        "Value": [0.8190, 0.9673, 0.8424, 0.9005, 0.99995, 0.8575]
                    }

                    # Create a DataFrame
                    df_metrics = pd.DataFrame(metrics)

                    # Style the DataFrame for better visual presentation
                    styled_df = df_metrics.style.background_gradient(cmap='Blues', subset=['Value']) \
                        .format({"Value": "{:.4f}"}) \
                        .set_table_styles([
                        {'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]}
                    ]) \
                        .set_properties(**{
                        'font-size': '20px',
                        'text-align': 'center',
                        'border-color': 'black',
                    })

                    # Afficher le tableau des métriques avec un titre
                    st.write("### 📊 Metrics Overview")

                    # Utiliser les colonnes pour ajuster la largeur
                    col1, col2 = st.columns([4, 1])  # La colonne centrale est deux fois plus large
                    with col1:
                        st.dataframe(styled_df, use_container_width=True)



            elif selected_sub == "Tester ou Prédire":
                if selected_main == "Heart":
                    st.write("Tester ou prédire le modèle pour les maladies cardiaques.")
                elif selected_main == "Lung Cancer":


                    # Charger le modèle

                    def load_model():
                        path_model = "epoch=27-step=48580.ckpt"  # Chemin vers le modèle entraîné
                        model = TumorSegmentation.load_from_checkpoint(path_model)
                        model.eval()
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model.to(device)
                        return model, device

                    # Fonction pour prédire le masque
                    def predict_mask(image_path):
                        image = np.load(image_path)
                        image = cv2.resize(image, (256, 256))  # Redimensionner si nécessaire
                        image_tensor = torch.tensor(image).float().to(device).unsqueeze(0).unsqueeze(0)

                        with torch.no_grad():
                            pred_mask = torch.sigmoid(model(image_tensor))
                            pred_mask = pred_mask[0][0].cpu().numpy()
                            pred_mask_binary = pred_mask > 0.5  # Binariser la prédiction

                        return image, pred_mask_binary

                    # Fonction pour lire et prédire à partir d'un fichier .nii.gz
                    def read_nifti(file):
                        file.seek(0)  # Réinitialiser le pointeur de fichier pour la lecture
                        nifti_data = nib.load(io.BytesIO(file.read())).get_fdata() / 3071
                        return nifti_data

                    # Chargement du modèle
                    model, device = load_model()


                    # Partie pour charger le fichier NumPy (.npy)
                    st.subheader("🎯 Charger un fichier NumPy (.npy)")
                    uploaded_npy = st.file_uploader("Choisissez un fichier NumPy", type=['npy'])

                    if uploaded_npy is not None:
                        try:
                            # Prédire et générer les images
                            image, pred_mask_binary = predict_mask(uploaded_npy)

                            # Affichage des images en une seule ligne avec trois colonnes
                            col1, col2, col3 = st.columns(3)

                            # Image Normale avec cmap="bone"
                            with col1:
                                fig, ax = plt.subplots()
                                ax.imshow(image, cmap="bone")
                                ax.axis('off')
                                st.pyplot(fig, bbox_inches='tight', pad_inches=0)
                                st.markdown("<p style='text-align: center;'>Image Normale</p>", unsafe_allow_html=True)

                            # Masque Prédit avec cmap="gray"
                            with col2:
                                fig, ax = plt.subplots()
                                ax.imshow(pred_mask_binary, cmap="gray")
                                ax.axis('off')
                                st.pyplot(fig, bbox_inches='tight', pad_inches=0)
                                st.markdown("<p style='text-align: center;'>Le Masque</p>", unsafe_allow_html=True)

                            # Superposition de l'image normale et du masque prédit
                            with col3:
                                contours = measure.find_contours(pred_mask_binary, 0.5)
                                fig, ax = plt.subplots()
                                ax.imshow(image, cmap="bone")
                                ax.imshow(np.ma.masked_where(pred_mask_binary == 0, pred_mask_binary), alpha=0.5,
                                          cmap="autumn")
                                for contour in contours:
                                    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red', alpha=0.8)
                                ax.axis('off')
                                st.pyplot(fig, bbox_inches='tight', pad_inches=0)
                                st.markdown("<p style='text-align: center;'>Image Superposée</p>",
                                            unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Erreur lors de la prédiction : {str(e)}")

                    st.markdown("---")

                    # Partie pour charger le fichier NIfTI (.nii.gz)
                    st.subheader("🔮 Charger un fichier NIfTI (.nii.gz)")
                    uploaded_file = st.file_uploader("Choisissez un fichier NIfTI (.nii.gz)", type=['nii', 'nii.gz'])

                    if uploaded_file is not None:
                        # Lire le fichier NIfTI
                        try:
                            # Créer un fichier temporaire avec l'extension correcte
                            suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                                temp_file.write(uploaded_file.read())
                                temp_path = temp_file.name

                            # Charger le fichier avec nibabel
                            ct_scan = nib.load(temp_path).get_fdata() / 3071  # Normalisation des valeurs du scan
                            ct_scan = ct_scan[:, :, 30:]  # Optionnel : sélectionner une tranche spécifique

                            # Variables pour stocker les prédictions et les images CT
                            segmentation, scan = [], []

                            # Boucle pour redimensionner et faire les prédictions sur chaque tranche
                            for i in range(ct_scan.shape[-1]):
                                slice_ = ct_scan[:, :, i]  # Extraire une tranche
                                slice_resized = cv2.resize(slice_, (256, 256))  # Redimensionner
                                scan.append(slice_resized)  # Ajouter l'image CT à la liste

                                slice_tensor = torch.tensor(slice_resized).unsqueeze(0).unsqueeze(0).float().to(
                                    device)  # Préparer pour le modèle

                                # Faire la prédiction avec le modèle
                                with torch.no_grad():
                                    pred = model(slice_tensor)[0][0].cpu()

                                pred_binary = pred > 0.5  # Binarisation du masque (seuil à 0.5)
                                segmentation.append(pred_binary.numpy())  # Ajouter le masque à la liste

                            # Création de l'animation avec les images et les masques
                            fig = plt.figure(figsize=(4, 4))  # Taille de la figure
                            camera = Camera(fig)

                            # Boucle pour afficher chaque image et le masque correspondant
                            for i in range(0, len(scan),
                                           2):  # Afficher toutes les 2 tranches pour accélérer l'animation
                                plt.imshow(scan[i], cmap="bone")  # Image CT-scan
                                mask = np.ma.masked_where(segmentation[i] == 0,
                                                          segmentation[i])  # Masquer les zones sans prédiction
                                plt.imshow(mask, alpha=0.5, cmap="autumn")  # Superposer le masque

                                # Trouver les contours dans le masque binaire
                                contours = measure.find_contours(segmentation[i], level=0.5)
                                # Ajouter les contours en rouge
                                for contour in contours:
                                    plt.plot(contour[:, 1], contour[:, 0], color='red',
                                             linewidth=2)  # Tracer les contours en rouge

                                plt.axis("off")  # Supprimer les axes
                                plt.subplots_adjust(left=0, right=1, top=1,
                                                    bottom=0)  # Ajuster les marges pour supprimer les bordures blanches
                                plt.tight_layout(pad=0)  # Ajuster les espaces pour supprimer les bordures blanches
                                camera.snap()

                            # Générer l'animation
                            animation = camera.animate(interval=50)  # Intervalle de 50ms entre les frames
                            plt.close()

                            # Sauvegarder la vidéo dans un fichier temporaire
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
                                animation.save(video_file.name)  # Enregistrer la vidéo
                                video_path = video_file.name

                            # Lire le contenu du fichier pour l'encodage en base64
                            with open(video_path, "rb") as video_file:
                                video_data = video_file.read()
                                video_base64 = base64.b64encode(video_data).decode("utf-8")

                            # Créer un lecteur vidéo avec la taille personnalisée via HTML
                            # Créer un lecteur vidéo centré avec la taille personnalisée via HTML et du CSS
                            video_html = f"""
                                <div style="display: flex; justify-content: center; align-items: center;">
                                    <video width="350" controls loop>
                                        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                                        Your browser does not support the video tag.
                                    </video>
                                </div>
                            """
                            st.markdown(video_html, unsafe_allow_html=True)


                        except Exception as e:
                            st.error(f"Erreur lors du chargement ou du traitement du fichier : {e}")

                    # Footer

    st.markdown("<footer style='text-align: center; margin-top: 50px;color:#828181'>"
                "Développé  par  laaouissi  azdine  "
                "</footer>", unsafe_allow_html=True)

# Appeler la fonction pour afficher le contenu
display_content()
