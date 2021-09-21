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

# body
st.write('원하는 이미지를 대상으로 다양한 실험을 편리하게 해볼 수 있습니다.')

uploaded_files = st.file_uploader('Please upload an image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
uploaded_files_list = list()

for uploaded_file in uploaded_files:
    log: str = f"id:{uploaded_file.id}, name:{uploaded_file.name}"
    st.write(log)
    uploaded_files_list.append(log)
    original_image = Image.open(uploaded_file)
    st.image(original_image)

# sidebar
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
        with st.form(key="parameters"):
            content_image = st.selectbox("Which images is content?", uploaded_files_list)
            style_image = st.selectbox("Which images is style?", uploaded_files_list)
            alpha = st.slider("Value of alpha:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            parameters_submitted = st.form_submit_button("Submit")

    st.header("Layout configuration")
    with st.form(key="grid_reset"):
        n_photos = st.slider("Number of images:", min_value=1, max_value=4, value=1)
        n_cols = st.number_input("Number of columns", min_value=1, max_value=4, value=1)
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success(n_photos)
            st.write("slider", n_photos, "checkbox", n_cols)

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
