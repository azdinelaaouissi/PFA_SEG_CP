#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# import streamlit_image_comparison as sic
# from skimage import measure
#
# # Charger l'image et le masque depuis les fichiers .npy
# slice = np.load("124.npy")
# mask = np.load("124 (1).npy")
#
# # Créer la première image (normale)
# fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
# ax.imshow(slice, cmap="bone")
# plt.axis('off')
# plt.savefig("lung_image.png", bbox_inches='tight', pad_inches=0)
# plt.close(fig)
#
# # Détecter les contours du masque
# contours = measure.find_contours(mask, 0.5)
#
# # Créer la deuxième image avec le masque et les contours
# fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
# ax.imshow(slice, cmap="bone")
# mask_ = np.ma.masked_where(mask == 0, mask)
# ax.imshow(mask_, alpha=0.5, cmap="autumn")  # Superposition avec transparence
#
# # Ajouter les contours en rouge
# for contour in contours:
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
#
# plt.axis('off')
# plt.savefig("lung_image_with_mask.png", bbox_inc    hes='tight', pad_inches=0)
# plt.close(fig)
#
# # Utiliser streamlit_image_comparison pour afficher les deux images
# st.header("Comparaison de l'image normale et superposée avec contours")
# sic.image_comparison(
#     img1="lung_image.png",
#     img2="lung_image_with_mask.png",
#     label1="Image Normale",
#     label2="Image Superposée avec Masque et Contours",
#     width=400
# )



# Charger les fichiers NIfTI
#
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# from celluloid import Camera
# import streamlit as st
# import io
# from PIL import Image
#
#
#
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
# from celluloid import Camera
# import streamlit as st
# import io
# from PIL import Image
#
# sample_patient_path = 'Task06_Lung/imagesTr/lung_046.nii.gz'
# sample_patient_label = 'Task06_Lung/labelsTr/lung_046.nii.gz'
# data = nib.load(sample_patient_path)
# label = nib.load(sample_patient_label)
#
# ct = data.get_fdata()
# mask = label.get_fdata()
#
# # Créer la figure et la caméra
# fig = plt.figure()
# camera = Camera(fig)
#
# # Boucle pour créer l'animation
# for i in range(0, ct.shape[2], 2):
#     plt.imshow(ct[:, :, i], cmap="bone")
#     mask_ = np.ma.masked_where(mask[:, :, i] == 0, mask[:, :, i])
#     plt.imshow(mask_, alpha=0.5, cmap="autumn")
#     camera.snap()
#
# # Créer l'animation
# animation = camera.animate(interval=50)
# plt.close()  # Close the plot to prevent display issues
#
# # Save the animation to a GIF file
# gif_path = 'animation.gif'
# animation.save(gif_path, writer='imagemagick')  # Save GIF using ImageMagick writer
#
# # Read the GIF file into memory
# with open(gif_path, 'rb') as f:
#     gif_data = f.read()
#
# # Display the GIF in Streamlit
# st.write("### Lung Cancer Segmentation Animation")
# st.image(gif_data)  # Display the GIF without specifying the format

#=============================================================================
#Task06_Lung/imagesTr/lung_015.nii.gz
import streamlit as st
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import load_img
from io import BytesIO
import matplotlib.pyplot as plt

# Chemins des fichiers locaux
nii_image_path = "Task06_Lung/imagesTr/lung_054.nii.gz"
nii_mask_path = "Task06_Lung/labelsTr/lung_054.nii.gz"

# Chargement des fichiers NIfTI
try:
    img = nib.load(nii_image_path)
    mask = nib.load(nii_mask_path)
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers NIfTI : {e}")
    st.stop()

# Convertir l'image en un format compatible pour affichage dans Streamlit
def get_img_data(img_path):
    img = load_img(img_path)
    return img

# Créer une figure avec matplotlib pour utiliser avec Streamlit
def plot_img_and_mask(image, mask):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plotting.plot_img(image, display_mode='ortho', axes=axes[0], title='Image')
    plotting.plot_img(mask, display_mode='ortho', axes=axes[1], title='Mask', cmap='autumn')
    plt.tight_layout()
    return fig

# Afficher les images dans Streamlit
st.title("Visualisation de l'image NIfTI et du masque")

# Afficher l'image originale
st.subheader("Image NIfTI")
image = get_img_data(nii_image_path)
mask = get_img_data(nii_mask_path)

fig = plot_img_and_mask(image, mask)

# Convertir la figure matplotlib en image Streamlit
buf = BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
st.image(buf, caption='Image et Masque', use_column_width=True)

plt.close(fig)
===========================
import streamlit as st
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import load_img
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Paths to the NIfTI files
nii_image_path = "Task06_Lung/imagesTr/lung_054.nii.gz"
nii_mask_path = "Task06_Lung/labelsTr/lung_054.nii.gz"

# Load the NIfTI files
try:
    img = nib.load(nii_image_path)
    mask = nib.load(nii_mask_path)
except Exception as e:
    st.error(f"Error loading NIfTI files: {e}")
    st.stop()

# Function to get image data
def get_img_data(img_path):
    return load_img(img_path)

# Function to plot a single NIfTI image
def plot_single_img(image):
    fig, ax = plt.subplots(figsize=(6, 6))
    plotting.plot_img(image, display_mode='ortho', axes=ax)
    plt.tight_layout()
    return fig

# Function to convert matplotlib figure to bytes
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return buf.read()

# Function to convert image bytes to base64
def bytes_to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode('utf-8')

# Streamlit interface
st.title("NIfTI Image and Mask Visualization")

# Define the target size for the images
target_size = (500, 400)  # Adjust the size as needed

# Generate the image and mask
image = get_img_data(nii_image_path)
fig_image = plot_single_img(image)
image_bytes = fig_to_bytes(fig_image)

mask = get_img_data(nii_mask_path)
fig_mask = plot_single_img(mask)
mask_bytes = fig_to_bytes(fig_mask)

# Convert image bytes to base64
image_base64 = bytes_to_base64(image_bytes)
mask_base64 = bytes_to_base64(mask_bytes)

# HTML for displaying images side by side with space in between
html = f"""
    <div style="display: flex; align-items: center;">
        <div style="margin-right: 20px; width: {target_size[0]}px;">
            <h3>NIfTI Image</h3>
            <img src="data:image/png;base64,{image_base64}" style="width: 100%; height: auto;">
        </div>
        <div style="margin-left: 20px; width: {target_size[0]}px;">
            <h3>NIfTI Mask</h3>
            <img src="data:image/png;base64,{mask_base64}" style="width: 100%; height: auto;">
        </div>
    </div>
"""

st.markdown(html, unsafe_allow_html=True)
