import streamlit as st
from PIL import Image, ImageOps

MAGE_EMOJI_URL = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/twitter/259/mage_1f9d9.png"

st.set_page_config(page_title="Vision intelligence GUI", page_icon=MAGE_EMOJI_URL)

# Display header.
st.markdown("<br>", unsafe_allow_html=True)
st.image(MAGE_EMOJI_URL, width=80)


st.title('Vision intelligence GUI✨')
"""
[![Github](http://img.shields.io/badge/-Github-black?style=flat-square&logo=github&link=https://github.com/takhyun12?tab=repositories)](https://github.com/takhyun12?tab=repositories)
&nbsp[![Linkedin](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/tackhyun-jung-a248941a8/)](https://www.linkedin.com/in/tackhyun-jung-a248941a8)
&nbsp[![Gmail](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:takhyun12@gmail.com)](mailto:takhyun12@gmail.com)
"""

st.write('Streamlit! 이 글은 튜토리얼 입니다.')

file = st.file_uploader('Please upload an image', type=['jpg', 'jpeg', 'png'])
if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
