import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import os
import gc

# 1. 페이지 설정: 사이드바 항상 펼침 고정
st.set_page_config(
    page_title="2023 AI Brand Insights", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded" 
)

# 2. CSS 최적화: Title/Date 한 줄 표시
st.markdown("""
    <style>
    [data-testid="stDataFrame"] td:nth-child(2), 
    [data-testid="stDataFrame"] td:nth-child(4) {
        white-space: nowrap !important;
    }
    section[data-testid="stSidebar"] {
        min-width: 280px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. AI 모델 로드 (메모리 최적화)
@st.cache_resource
def load_sentiment_model():
    try:
        return pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
    except:
        return None

# 4. 데이터 로드 및 정렬 함수
def load_and_sort_by_entry(filename):
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
        df['entry_num'] = df['Title'].apply(lambda x: int(re.search(r'#(\d+)', str(x)).group(1)) if re.search(r'#(\d+)', str(x)) else 999)
        df = df.sort_values(by='entry_num').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df.drop(columns=['entry_num'])
        df.index = df.index + 1 
        return df
    except:
        return pd.DataFrame()

# 5. 사이드바 고정 영역: 디자인은 항상 유지됨
with st.sidebar:
    st.title("🚀 Navigate & Analysis")
    st.markdown("---")
    
    # Quick Start 안내창 항상 표시
    st.markdown("""
        <div style="background-color: #e1f5fe; padding: 15px; border-radius: 10px; border-left: 5px solid #03a9f4; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.95em; color: #01579b; font-weight: bold;">💡 Quick Start</p>
            <p style="margin: 0; font-size: 0.85em; color: #0277bd;">Explore data insights. 'Month Filter' applies to AI Review Analysis only.</p>
        </div>
        """, unsafe_allow_html=True)

    # 메뉴 선택
    menu = st.radio(
        "Go to", 
        ["📦 Product Insights", "💬 Testimonial Stories", "⭐ Review Analytics"], 
        index=0
    )
    
    st.markdown("---")
    
    # [디자인 고정] 월 선택 슬라이더는 항상 보임
    st.subheader("🗓️ Analysis Filter")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sel_month = st.select_slider("Select Month", options=range(1, 13), format_func=lambda x: month_names[x-1])

# 6. 메인 콘텐츠 영역
if menu == "📦 Product Insights":
    st.title("📦 Product Insights")
    df = load_and_sort_by_entry("products.csv") #
    if not df.empty:
        # [기능 분리] 필터링 없이 전체 리스트 출력
        st.write(f"Showing all **{len(df)}** products in numerical order.")
        st.dataframe(df[['Title', 'Text', 'Date']], use_container_width=True)
    else:
        st.warning("products.csv 파일을 찾을 수 없습니다.")

elif menu == "💬 Testimonial Stories":
    st.title("💬 Testimonial Stories")
    df = load_and_sort_by_entry("testimonials.csv") #
    if not df.empty:
        # [기능 분리] 필터링 없이 전체 리스트 출력
        st.write(f"Showing all **{len(df)}** user testimonials.")
        st.dataframe(df[['Title', 'Text', 'Date']], use_container_width=True)
    else:
        st.warning("testimonials.csv 파일을 찾을 수 없습니다.")

else: # ⭐ Review Analytics
    st.title("⭐ Deep Learning Review Analysis")
    df = load_and_sort_by_entry("reviews.csv") #
    if not df.empty:
        # [기능 연결] 오직 이 메뉴에서만 사이드바의 sel_month를 사용하여 필터링 수행
        df['Date_dt'] = pd.to_datetime(df['Date'])
        filtered = df[df['Date_dt'].dt.month == sel_month].copy()
        
        if not filtered.empty:
            with st.spinner(f'Analyzing reviews for {month_names[sel_month-1]}...'):
                analyzer = load_sentiment_model()
                if analyzer:
                    res = analyzer(filtered['Text'].tolist())
                    filtered['Sentiment'] = [r['label'] for r in res]
                    gc.collect() # 메모리 관리
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Sentiment Distribution")
                fig = px.bar(filtered['Sentiment'].value_counts().reset_index(), x='Sentiment', y='count', color='Sentiment',
                             color_discrete_map={'POSITIVE': '#00b894', 'NEGATIVE': '#ff7675'})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("☁️ Word Cloud")
                wc = WordCloud(background_color="white", stopwords=STOPWORDS, width=800, height=450).generate(" ".join(filtered['Text']))
                fig_wc, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
                st.pyplot(fig_wc)
        else:
            st.warning(f"{month_names[sel_month-1]}월에는 분석할 리뷰 데이터가 없습니다.")
    else:
        st.error("reviews.csv 파일을 찾을 수 없습니다.")