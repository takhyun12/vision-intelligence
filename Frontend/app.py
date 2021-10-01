__version__ = '1.0.0'

import cv2
import numpy as np
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageOps
from models.adain import AdaIN_v1

# configuration
task_dict = {
    "Face Alignment": {
        "FFHQ-Alignment": {
            "Public dataset": "location"
        },
        "PRNET": {
            "Public dataset": "location"
        },
    },
    "Style Transfer": {
        "AdaIN": {
            "VGG Encoder": "location"
        },
    },
    "Face Generator": {
        "starGAN_v1": {
            "CelebA-HQ": "location"
        },
        "starGAN_v2": {
            "CelebA-HQ": "location",
            "AFHQ(Animal Faces-HQ)": "location",
            "AFHQ_v2": "location"
        },
        "styleCLIP": {
            "Public dataset": "location"
        },
        "styleGAN_v2": {
            "Public dataset": "location"
        },
    }
}


# resource => 객체화
MAGE_EMOJI_URL = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/twitter/259/mage_1f9d9.png"
GIT_EMOJI_URL = "http://img.shields.io/badge/-Github-black?style=flat-square&logo=github&link=https://github.com/takhyun12?tab=repositories"
LINKEDIN_EMOJI_URL = "https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/tackhyun-jung-a248941a8/"
GMAIL_EMOJI_URL = "https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:takhyun12@gmail.com"

# browser header
st.set_page_config(page_title="Vision intelligence GUI", page_icon=MAGE_EMOJI_URL)
st.markdown("<br>", unsafe_allow_html=True)

# header
st.image(MAGE_EMOJI_URL, width=80)
st.title('GUI for Vision intelligence research✨')
f"""
[![Github]({GIT_EMOJI_URL})](https://github.com/takhyun12?tab=repositories)
&nbsp[![Linkedin]({LINKEDIN_EMOJI_URL})](https://www.linkedin.com/in/tackhyun-jung-a248941a8)
&nbsp[![Gmail]({GMAIL_EMOJI_URL})](mailto:takhyun12@gmail.com)
"""

body_form = st.form("body_form")
body_submitted = body_form.form_submit_button("Save")

# sidebar => 객체화
with st.sidebar:
    st.info("🎈 **NEW:** Add your own code template to this site! [Guide](https://github.com/jrieke/traingenerator#adding-new-templates)")

    st.write("## Task")
    task = st.selectbox("Which problem do you want to solve?", list(task_dict.keys()))

    st.write("## Model")
    model = st.selectbox(
        "Which model?", list(task_dict[task].keys())
    )

    st.write("## Dataset")
    dataset = st.selectbox("Which data do you want to use?", list(task_dict[task][model].keys()))

    if task == "Style Transfer":
        st.header("Parameters")

        # content image
        content_image_path = st.file_uploader('Which images is content?', type=['jpg', 'jpeg', 'png'])

        content_aspect = st.sidebar.radio(label="Content Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {"1:1": (1, 1),
                       "16:9": (16, 9),
                       "4:3": (4, 3),
                       "2:3": (2, 3),
                       "Free": None}
        content_aspect_ratio = aspect_dict[content_aspect]

        content_degree = st.sidebar.slider('Content rotate', -45, 45, 0, 1)

        body_form.header("Selected Images")
        content_column, style_column = body_form.columns(2)
        if content_image_path:
            file_bytes = np.asarray(bytearray(content_image_path.read()), dtype=np.uint8)
            content_image = cv2.imdecode(file_bytes, 1)

            (h, w) = content_image.shape[:2]
            (cX, cY) = (w / 2, h / 2)

            M = cv2.getRotationMatrix2D((cX, cY), content_degree, 1.0)
            rotated_content_image = cv2.warpAffine(content_image, M, (w, h))

            color_coverted = cv2.cvtColor(rotated_content_image, cv2.COLOR_BGR2RGB)
            content_image = Image.fromarray(color_coverted)

            with body_form:
                cropped_content_image = st_cropper(content_image,
                                         realtime_update=True,
                                         aspect_ratio=content_aspect_ratio)

                with content_column:
                    st.write("content image thumbnail")
                    st.image(cropped_content_image, width=256)

        st.markdown("---")

        style_image_path = st.file_uploader('Which images is style?', type=['jpg', 'jpeg', 'png'])

        style_aspect = st.sidebar.radio(label="Style Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {"1:1": (1, 1),
                       "16:9": (16, 9),
                       "4:3": (4, 3),
                       "2:3": (2, 3),
                       "Free": None}
        style_aspect_ratio = aspect_dict[style_aspect]

        style_degree = st.sidebar.slider('Style rotate', -45, 45, 0, 1)

        if style_image_path:
            file_bytes = np.asarray(bytearray(style_image_path.read()), dtype=np.uint8)
            style_image = cv2.imdecode(file_bytes, 1)

            (h, w) = style_image.shape[:2]
            (cX, cY) = (w / 2, h / 2)

            M = cv2.getRotationMatrix2D((cX, cY), style_degree, 1.0)
            rotated_style_image = cv2.warpAffine(style_image, M, (w, h))

            color_coverted = cv2.cvtColor(rotated_style_image, cv2.COLOR_BGR2RGB)
            style_image = Image.fromarray(color_coverted)

            with body_form:
                cropped_style_image = st_cropper(style_image,
                                         realtime_update=True,
                                         aspect_ratio=style_aspect_ratio)

                with style_column:
                    st.write("style image thumbnail")
                    st.image(cropped_style_image, width=256)

        st.markdown("---")

        alpha = st.slider("Value of alpha:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

        if content_image_path and style_image_path:
            assert (0.0 <= alpha <= 1.0)

            adain = AdaIN_v1(content_image=content_image_path, style_image=style_image_path, alpha=alpha)
            result_image_path = adain.style_transfer()
            result_image = Image.open(result_image_path)
            body_form.image(result_image)

    st.header("Layout configuration")
    n_photos = st.slider("Number of images:", min_value=1, max_value=4, value=1)
    n_cols = st.number_input("Number of columns", min_value=1, max_value=4, value=1)

    st.write("## Training")
    gpu_check = st.checkbox('Use GPU if available')
    if gpu_check:
        st.info('Great!')

    save_checkpoint = st.checkbox('Save model checkpoint each epoch')
    if save_checkpoint:
        st.info('Great!')

    st.error(
        "Found a bug? [Report it](https://github.com/jrieke/traingenerator/issues) 🐛"
    )

# ===========================