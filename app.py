import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud
from datetime import datetime

# 1. 페이지 설정
st.set_page_config(page_title="Brand Reputation 2023", layout="wide")

# 2. 데이터 및 모델 로드 (캐싱을 통해 속도 향상)
@st.cache_data
def load_data():
    # scrape.py에서 만든 파일 읽기
    df = pd.read_csv("scraped_reviews.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_sentiment_model():
    # 과제 지정 모델: distilbert-base-uncased-finetuned-sst-2-english
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 데이터 불러오기 시도
try:
    df = load_data()
    sentiment_model = load_sentiment_model()
except Exception as e:
    st.error(f"데이터를 불러오는 데 실패했습니다: {e}")
    st.info("먼저 'py scrape.py'를 실행하여 csv 파일을 생성했는지 확인하세요.")
    st.stop()

# --- 3. 사이드바 내비게이션 ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Products", "Testimonials", "Reviews"])

# --- 4. 메인 화면 구성 ---

if page == "Products":
    st.title("🎁 Our Products")
    st.write("2023년 주요 제품 목록입니다.")
    st.dataframe(df[['Title', 'Date']], use_container_width=True)

elif page == "Testimonials":
    st.title("💬 Testimonials")
    st.write("고객들의 추천사 데이터를 확인하세요.")
    st.table(df[['Title', 'Text']])

elif page == "Reviews":
    st.title("⭐ Review Sentiment Analysis")
    st.markdown("---")

    # [요건] 2023년 월별 선택 슬라이더
    st.subheader("📅 Select Month in 2023")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    selected_month_name = st.select_slider("분석할 달을 선택하세요", options=months)
    
    # 월 이름 숫자로 변환
    month_num = months.index(selected_month_name) + 1

    # [요건] 선택한 월의 데이터만 필터링
    filtered_df = df[
        (df['Date'].dt.year == 2023) & 
        (df['Date'].dt.month == month_num)
    ].copy()

    if not filtered_df.empty:
        # [요건] Hugging Face 모델로 감성 분석 수행
        with st.spinner('AI 모델이 리뷰를 분석 중입니다...'):
            texts = filtered_df['Text'].tolist()
            predictions = sentiment_model(texts)
            
            filtered_df['Sentiment'] = [p['label'] for p in predictions]
            filtered_df['Confidence'] = [round(p['score'], 4) for p in predictions]

        # 데이터 프레임 출력
        st.write(f"### {selected_month_name} 2023 리뷰 리스트")
        st.dataframe(filtered_df, use_container_width=True)

        # 컬럼 레이아웃 (차트 나란히 배치)
        col1, col2 = st.columns(2)

        with col1:
            # [요건] 시각화 - 긍정/부정 막대 그래프
            st.write("#### 📊 Sentiment Count")
            sentiment_counts = filtered_df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
            # [요건] 평균 신뢰도 점수 표시
            avg_conf = filtered_df['Confidence'].mean()
            st.metric("Model Confidence Score (Avg)", f"{avg_conf:.2%}")

        with col2:
            # [보너스] 워드클라우드 시각화
            st.write("#### ☁️ Review Word Cloud")
            all_text = " ".join(filtered_df['Text'].tolist())
            if all_text.strip():
                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(all_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

    else:
        st.warning(f"데이터가 없습니다: {selected_month_name} 2023에 등록된 리뷰가 없습니다.")