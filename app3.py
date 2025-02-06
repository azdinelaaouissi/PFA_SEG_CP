import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit_image_comparison as sic
from skimage import measure

# Function to process and save images from npy files
def process_and_save_images(slice_path, mask_path, normal_img_path, mask_img_path):
    slice_img = np.load(slice_path)
    mask_img = np.load(mask_path)

    # Create the normal image
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.imshow(slice_img, cmap="bone")
    plt.axis('off')
    plt.savefig(normal_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Detect contours
    contours = measure.find_contours(mask_img, 0.5)

    # Create the image with mask and contours
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.imshow(slice_img, cmap="bone")
    mask_ = np.ma.masked_where(mask_img == 0, mask_img)
    ax.imshow(mask_, alpha=0.5, cmap="autumn")

    # Add the contours in red
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

    plt.axis('off')
    plt.savefig(mask_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Process images and save them
process_and_save_images("124.npy", "124 (1).npy", "image1_normal.png", "image1_with_mask.png")
process_and_save_images("125.npy", "125 (1).npy", "image2_normal.png", "image2_with_mask.png")
process_and_save_images("123.npy", "123 (1).npy", "image3_normal.png", "image3_with_mask.png")

# Set page configuration for wide layout
st.set_page_config(layout='wide')

# Main title
st.title("Projet de Détection du Cancer du Poumon")

# Subheader for comparison
st.subheader("Comparaison des Images Segmentées")

# Adding custom CSS for better image layout
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
    </style>
    """,
    unsafe_allow_html=True
)

# Creating a layout using columns with adjustable spacing
col1, col_space1, col2, col_space2, col3 = st.columns([1, 0.1, 1, 0.1, 1])

# Image 1 comparison
with col1:
    sic.image_comparison(
        img1="image1_normal.png",
        img2="image1_with_mask.png",
        label1="Image 1 Normale",
        label2="Image 1 Superposée avec Masque et Contours",
        width=300
    )

# Spacer using HTML
with col_space1:
    st.markdown("<div style='width: 10px;'></div>", unsafe_allow_html=True)

# Image 2 comparison
with col2:
    sic.image_comparison(
        img1="image2_normal.png",
        img2="image2_with_mask.png",
        label1="Image 2 Normale",
        label2="Image 2 Superposée avec Masque et Contours",
        width=300
    )

# Spacer using HTML
with col_space2:
    st.markdown("<div style='width: 10px;'></div>", unsafe_allow_html=True)

# Image 3 comparison
with col3:
    sic.image_comparison(
        img1="image3_normal.png",
        img2="image3_with_mask.png",
        label1="Image 3 Normale",
        label2="Image 3 Superposée avec Masque et Contours",
        width=300
    )
