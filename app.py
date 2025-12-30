import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re, os, gc

# 페이지 설정
st.set_page_config(page_title="2023 AI Brand Insights", layout="wide")

# [핵심] 환경에 따른 모델 자동 선택 로직
@st.cache_resource
def load_sentiment_model():
    # Render 서버인지 로컬 PC인지 확인 (Render는 고유 환경변수를 가짐)
    is_render = "RENDER" in os.environ
    
    try:
        if is_render:
            # Render 서버: 503 에러 방지를 위해 초경량 모델 사용
            model_name = "prajjwal1/bert-tiny"
        else:
            # 로컬 PC: 정확한 분석(Negative 추출)을 위해 고성능 모델 사용
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            
        return pipeline("sentiment-analysis", model=model_name, device=-1)
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None

def load_data(filename):
    if not os.path.exists(filename): return pd.DataFrame()
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
        # 데이터가 비어있지 않다면 일련번호 1부터 시작
        df.index = df.index + 1
        return df
    except:
        return pd.DataFrame()

# 사이드바 구성
with st.sidebar:
    st.title("🚀 Navigate & Analysis")
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #eee; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.9em; color: #444; line-height: 1.4;">
                🔎 Explore what's happening with your brand.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    menu = st.radio("Go to", ["📦 Product Insights", "💬 Testimonial Stories", "⭐ Review Analytics"], index=2)
    st.markdown("---")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sel_month = st.select_slider("Select Month", options=range(1, 13), format_func=lambda x: month_names[x-1])

if menu == "⭐ Review Analytics":
    st.title("⭐ Deep Learning Review Analysis")
    df = load_data("reviews.csv")
    
    if not df.empty:
        # 날짜 필터링 로직
        df['Date_dt'] = pd.to_datetime(df['Date'])
        filtered = df[df['Date_dt'].dt.month == sel_month].copy()
        
        if not filtered.empty:
            with st.spinner('AI 분석 중... (로컬은 정밀 분석, 서버는 쾌속 분석)'):
                analyzer = load_sentiment_model()
                if analyzer:
                    results = analyzer(filtered['Text'].tolist())
                    # 모델마다 다른 라벨 형식을 POSITIVE/NEGATIVE로 통일
                    filtered['Sentiment'] = [
                        "POSITIVE" if r['label'] in ['LABEL_1', 'POSITIVE'] else "NEGATIVE" 
                        for r in results
                    ]
                    filtered['Confidence'] = [r['score'] for r in results]
                    gc.collect() # 메모리 관리

            # 결과 시각화
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Sentiment Distribution")
                chart_data = filtered.groupby('Sentiment')['Confidence'].agg(['count', 'mean']).reset_index()
                chart_data.columns = ['Sentiment', 'Review Count', 'Avg. Confidence']
                
                # 디자인 적용: Sentiment(Bold), Confidence(Bold)
                chart_data['Display Label'] = chart_data.apply(
                    lambda x: f"<span style='font-size:16px; font-weight:bold;'>{x['Sentiment']}</span><br>"
                              f"<span style='font-size:12px; font-weight:bold; color:#555;'>Avg. Confidence ({x['Avg. Confidence']:.4f})</span>", 
                    axis=1
                )
                
                fig = px.bar(chart_data, x='Display Label', y='Review Count', color='Sentiment',
                             color_discrete_map={'POSITIVE': '#00b894', 'NEGATIVE': '#ff7675'})
                fig.update_traces(texttemplate='%{y}', textposition='outside')
                fig.update_layout(xaxis_title="", yaxis_title="Review Count", xaxis={'tickangle': 0})
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("☁️ Word Cloud")
                text_data = " ".join(filtered['Text'])
                if text_data.strip():
                    wc = WordCloud(background_color="white", width=800, height=500).generate(text_data)
                    fig_wc, ax = plt.subplots(); ax.imshow(wc); ax.axis("off")
                    st.pyplot(fig_wc)
        else:
            st.warning(f"{month_names[sel_month-1]}월 리뷰가 없습니다.")
    else:
        st.error("reviews.csv 파일을 찾을 수 없습니다.")
else:
    st.title(menu)
    df = load_data("products.csv" if "Product" in menu else "testimonials.csv")
    st.dataframe(df, use_container_width=True)