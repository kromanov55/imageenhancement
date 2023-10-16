import base64
import time
import streamlit_antd_components as sac
import streamlit as st
from streamlit_image_comparison import image_comparison
import cv2
from cv2 import dnn_superres
import os
import numpy as np

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

set_background('back.png')

st.title("Улучшение изображений")

nav = sac.tabs([
    sac.TabsItem(label='Upscale', icon='file-arrow-up'),
    sac.TabsItem(label='Colorize', icon='brush'),
    sac.TabsItem(label='Crop', icon='crop'),
    sac.TabsItem(label='Filtering', icon='droplet'),
    sac.TabsItem(label='Cutout', icon='eraser-fill')
], format_func='title', align='center', grow=True)

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
ready = False
if uploaded_file is not None:
    with st.status("Делаем магию...", expanded=True) as status:
        st.write("Загружаем файл...")

        with open(os.path.join("tempDir/lowres", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        time.sleep(1)

        st.write("Применяем алгоритм...")
        n = f"tempDir/lowres/{uploaded_file.name}"
        # Read image
        image = cv2.imread(n)

        # Read the desired model
        path = "EDSR_x4.pb"
        sr.readModel(path)

        # Set the desired model and scale to get correct pre- and post-processing
        sr.setModel("edsr", 3)

        # Upscale the image
        result = sr.upsample(image)

        # Save the image
        cv2.imwrite(f"tempDir/highres/u{uploaded_file.name}", result)

        time.sleep(1)
        st.write("Дорабатываем изображение...")

        image_hr = cv2.imread(f'tempDir/highres/u{uploaded_file.name}')

        sharpened_pre = unsharp_mask(image_hr)

        sharpened = unsharp_mask(sharpened_pre)

        cv2.imwrite(f'tempDir/sharpened/s{uploaded_file.name}', sharpened)

        ready = True

        status.update(label="Обработка закончена!", state="complete", expanded=False)

    if ready == True:
        image_comparison(
            img1=f"tempDir/lowres/{uploaded_file.name}",
            img2=f"tempDir/sharpened/s{uploaded_file.name}",
        )

        with open(f"tempDir/sharpened/s{uploaded_file.name}", "rb") as file:
            btn = st.download_button(
                label="Скачать улучшенное изображение",
                data=file,
                file_name=f"tempDir/sharpened/s{uploaded_file.name}",
                #mime="image/png"
            )
