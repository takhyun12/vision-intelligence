import streamlit as st
from PIL import Image, ImageOps

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
for uploaded_file in uploaded_files:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    with col1:
        st.image(image)

    with col2:
        st.image(image)

# sidebar
with st.sidebar:
    st.header("Layout configuration")
    with st.form(key="grid_reset"):
        n_photos = st.slider("Number of images:", min_value=1, max_value=4, value=1)
        n_cols = st.number_input("Number of columns", min_value=1, max_value=4, value=1)
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success(n_photos)
            st.write("slider", n_photos, "checkbox", n_cols)



