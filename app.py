import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 이미지 분류기", page_icon="📷", layout="centered")

# 사이드바 설정
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    st.write("1. 이미지를 업로드하거나 카메라로 촬영하세요.")
    st.write("2. 웹 URL을 입력하여 이미지를 불러올 수도 있습니다.")
    st.write("3. AI가 이미지의 객체를 분석하고 분류 결과를 제공합니다.")
    st.markdown("---")
    st.write("💡 *신뢰도가 높을수록 정확도가 높습니다!*")

# 페이지 제목 및 설명
st.markdown("<h1>🦎<span style='font-size:24px;'>(사탄잎꼬리도마뱀붙이)</span> VS 🍂<span style='font-size:24px;'>(나뭇잎)</span></h1>", unsafe_allow_html=True)
st.subheader("📷 AI 기반 이미지 분류기")
st.subheader("이미지를 업로드하거나 카메라를 이용해 AI가 분류 결과를 알려줍니다.")
st.markdown("---")

# 모델 로드
model = load_model('keras_model.h5', compile=False)
class_names = open('labels.txt', 'r').readlines()

# 입력 방식 선택
input_method = st.radio("📥 이미지 입력 방식 선택", ["파일 업로드", "카메라 사용", "이미지 URL 입력"], horizontal=True)

if input_method == "카메라 사용":
    img_file_buffer = st.camera_input("📷 정중앙에 사물을 위치하고 사진을 촬영하세요.")
elif input_method == "파일 업로드":
    img_file_buffer = st.file_uploader("📁 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
elif input_method == "이미지 URL 입력":
    img_url = st.text_input("🔗 이미지 URL을 입력하세요:")
    if img_url:
        try:
            response = requests.get(img_url)
            img_file_buffer = BytesIO(response.content)
        except:
            st.error("❌ 유효한 이미지 URL을 입력하세요.")
            img_file_buffer = None
else:
    img_file_buffer = None

# 이미지가 입력되었을 경우 처리
if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert('RGB')
    st.image(image, caption="📌 업로드된 이미지", use_container_width=True)

    # 이미지 전처리
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # 예측 수행
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    # 결과 출력
    st.markdown("---")
    st.subheader("🔍 예측 결과")
    result_container = st.container()
    with result_container:
        if confidence_score > 0.8:
            st.success(f"🎯 **분류 결과:** {class_name}")
        elif confidence_score > 0.5:
            st.warning(f"⚠️ **분류 결과:** {class_name}")
        else:
            st.error(f"❌ **분류 결과:** {class_name}")
    
    st.write(f"📊 **신뢰도:** {confidence_score:.2%}")
    
    # NanumGothic 폰트 적용
    font_path = 'C:/Users/jimin/AppData/Local/Microsoft/Windows/Fonts/NanumGothic.ttf'  # 시스템 폰트 경로 (리눅스 기준)
    font_prop = fm.FontProperties(fname=font_path, size=12)

    # 신뢰도 바 그래프 추가
    fig, ax = plt.subplots(figsize=(5, 1))
    ax.barh(["신뢰도"], [confidence_score], color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("신뢰도 점수", fontproperties=font_prop)
    ax.set_title("🔍 신뢰도 그래프", fontproperties=font_prop)
    ax.set_yticklabels(["신뢰도"], fontproperties=font_prop)  # y축 한글 적용

    st.pyplot(fig)