import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import os

# 1. 페이지 설정
st.set_page_config(page_title="2023 AI Brand Insights", layout="wide", page_icon="📈")

# 2. AI 모델 로드
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 3. Entry 번호순 정렬 함수
def load_and_sort_by_entry(filename):
    if not os.path.exists(filename):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filename)
        # Title에서 숫자를 추출하여 정렬 (예: Entry #10 -> 10)
        df['entry_num'] = df['Title'].apply(lambda x: int(re.search(r'#(\d+)', x).group(1)) if re.search(r'#(\d+)', x) else 0)
        
        # 숫자 순서대로 오름차순 정렬 (#1, #2, #3...)
        df = df.sort_values(by='entry_num').reset_index(drop=True)
        
        # 표시용 날짜 정리
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        df = df.drop(columns=['entry_num'])
        df.index = df.index + 1 
        return df
    except:
        return pd.DataFrame()

# 4. 사이드바 디자인
st.sidebar.title("🚀 Navigate & Analysis")
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="background-color: #e1f5fe; padding: 15px; border-radius: 10px; border-left: 5px solid #03a9f4; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 0.95em; color: #01579b; font-weight: bold;">💡 Quick Start</p>
        <p style="margin: 0; font-size: 0.85em; color: #0277bd;">Select a category to explore data & AI insights.</p>
    </div>
    """, unsafe_allow_html=True)

menu = st.sidebar.radio("Go to", ["📦 Product Insights", "💬 Testimonial Stories", "⭐ Review Analytics"], index=2)

# 5. 메인 콘텐츠
if menu == "⭐ Review Analytics":
    st.title("⭐ Deep Learning Review Analysis")
    df = load_and_sort_by_entry("reviews.csv")
    
    if not df.empty:
        # 월별 슬라이더 필터링
        df['Date_dt'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date_dt'].dt.month
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        sel_month = st.sidebar.select_slider("Select Month", options=range(1, 13), format_func=lambda x: month_names[x-1])
        
        filtered = df[df['Month'] == sel_month].copy()
        
        if not filtered.empty:
            with st.spinner('AI 분석 중...'):
                analyzer = load_sentiment_model()
                res = analyzer(filtered['Text'].tolist())
                filtered['Sentiment'] = [r['label'] for r in res]
            
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
            st.warning(f"{month_names[sel_month-1]}월에 해당하는 데이터가 없습니다.")
    else:
        st.error("⚠️ reviews.csv 파일을 찾을 수 없습니다. 수집 스크립트를 먼저 실행해주세요.")

elif menu == "📦 Product Insights":
    st.title("📦 Product Insights")
    df = load_and_sort_by_entry("products.csv")
    if not df.empty:
        st.write(f"Showing **{len(df)}** items sorted by **Entry Number (#1, #2...)**")
        st.table(df[['Title', 'Text', 'Date']])
    else:
        st.error("products.csv not found.")

else: # Testimonial Stories
    st.title("💬 Testimonial Stories")
    df = load_and_sort_by_entry("testimonials.csv")
    if not df.empty:
        st.write(f"Showing **{len(df)}** user stories sorted by **Entry Number (#1, #2...)**")
        st.table(df[['Title', 'Text', 'Date']])
    else:
        st.error("testimonials.csv not found.")