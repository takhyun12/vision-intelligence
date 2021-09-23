import streamlit as st
from PIL import Image, ImageOps

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

# browser header
st.set_page_config(page_title="Vision intelligence GUI", page_icon=MAGE_EMOJI_URL)
st.markdown("<br>", unsafe_allow_html=True)

# header
st.image(MAGE_EMOJI_URL, width=80)
st.title('GUI for Vision intelligence research✨')
"""
[![Github](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github&link=https://github.com/takhyun12?tab=repositories)](https://github.com/takhyun12?tab=repositories)
&nbsp[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/tackhyun-jung-a248941a8/)](https://www.linkedin.com/in/tackhyun-jung-a248941a8)
&nbsp[![Gmail](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:takhyun12@gmail.com)](mailto:takhyun12@gmail.com)
"""

body_form = st.form("body_form")

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

        content_image_path = st.file_uploader('Which images is content?', type=['jpg', 'jpeg', 'png'])
        if content_image_path:
            content_image = Image.open(content_image_path)
            body_form.image(content_image)

        style_image_path = st.file_uploader('Which images is style?', type=['jpg', 'jpeg', 'png'])
        if style_image_path:
            style_image = Image.open(style_image_path)
            body_form.image(style_image)

        alpha = st.slider("Value of alpha:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

        if content_image_path and style_image_path:
            st.info(content_image_path)
            result_image = Image.open(content_image_path)
            result_image = result_image.convert('L')
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


# body => 객체화
body_submitted = body_form.form_submit_button("Save")