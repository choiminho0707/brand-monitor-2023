import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import os
import gc

# 1. 페이지 설정 및 가독성 CSS
st.set_page_config(page_title="2023 AI Brand Insights", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>[data-testid='stDataFrame'] td:nth-child(2), [data-testid='stDataFrame'] td:nth-child(4) {white-space: nowrap !important;}</style>", unsafe_allow_html=True)

# [503 에러 방지] 모델 로드 최적화 (가장 안정적인 distilbert 사용)
@st.cache_resource
def load_sentiment_model():
    try:
        # 메모리 점유를 줄이기 위해 CPU 모드로 명시적 설정
        return pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            device=-1 
        )
    except:
        return None

def load_and_sort_by_entry(filename):
    if not os.path.exists(filename): return pd.DataFrame()
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
        df['entry_num'] = df['Title'].apply(lambda x: int(re.search(r'#(\d+)', str(x)).group(1)) if re.search(r'#(\d+)', str(x)) else 999)
        df = df.sort_values(by='entry_num').reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df.drop(columns=['entry_num'])
        # 일련번호 1부터 시작하도록 수정
        df.index = df.index + 1 
        return df
    except:
        return pd.DataFrame()

# 2. 사이드바 내비게이션 및 요청 문구 반영
with st.sidebar:
    st.title("🚀 Navigate & Analysis")
    
    # 요청하신 노멀하고 깔끔한 안내 문구 적용
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #eee;">
            <p style="margin: 0; font-size: 0.9em; color: #444; line-height: 1.4;">
                🔎 Explore what's happening with your brand.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("") 
    menu = st.radio("Go to", ["📦 Product Insights", "💬 Testimonial Stories", "⭐ Review Analytics"], index=2)
    st.markdown("---")
    
    # 날짜 필터 (Review Analytics 메뉴에서만 작동)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sel_month = st.select_slider("Select Month", options=range(1, 13), format_func=lambda x: month_names[x-1])

# 3. 메인 콘텐츠
if menu == "⭐ Review Analytics":
    st.title("⭐ Deep Learning Review Analysis")
    df = load_and_sort_by_entry("reviews.csv") 
    
    if not df.empty:
        df['Date_dt'] = pd.to_datetime(df['Date'])
        filtered = df[df['Date_dt'].dt.month == sel_month].copy()
        
        if not filtered.empty:
            with st.spinner(f'Analyzing {month_names[sel_month-1]} reviews...'):
                analyzer = load_sentiment_model()
                if analyzer:
                    # 분석 실행 및 메모리 관리
                    results = analyzer(filtered['Text'].tolist())
                    filtered['Sentiment'] = [r['label'] for r in results]
                    filtered['Confidence'] = [r['score'] for r in results]
                    
                    # 사용 직후 즉시 메모리 정리 (503 에러 핵심 방어)
                    del results
                    gc.collect() 

            # 결과 시각화 (X축 라벨 디자인 유지)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Sentiment Distribution")
                chart_data = filtered.groupby('Sentiment')['Confidence'].agg(['count', 'mean']).reset_index()
                chart_data.columns = ['Sentiment', 'Review Count', 'Avg. Confidence']
                
                chart_data['Display Label'] = chart_data.apply(
                    lambda x: f"<span style='font-size:18px; font-weight:bold;'>{x['Sentiment']}</span><br>"
                              f"<span style='font-size:14px; font-weight:bold; color:#444;'>Avg. Confidence ({x['Avg. Confidence']:.4f})</span>", 
                    axis=1
                )
                
                fig = px.bar(chart_data, x='Display Label', y='Review Count', color='Sentiment',
                             color_discrete_map={'POSITIVE': '#00b894', 'NEGATIVE': '#ff7675'})
                fig.update_traces(texttemplate='%{y}', textposition='outside')
                fig.update_layout(xaxis_title="", yaxis_title="Review Count", xaxis={'tickangle': 0})
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("☁️ Word Cloud")
                wc = WordCloud(background_color="white", width=800, height=450).generate(" ".join(filtered['Text']))
                fig_wc, ax = plt.subplots(); ax.imshow(wc); ax.axis("off")
                st.pyplot(fig_wc)
        else:
            st.warning(f"{month_names[sel_month-1]}월 데이터가 없습니다.")
    else:
        st.error("reviews.csv 파일을 찾을 수 없습니다.")

else: # Product & Testimonial (일련번호 1번부터 시작)
    st.title(f"{menu}")
    filename = "products.csv" if "Product" in menu else "testimonials.csv"
    df = load_and_sort_by_entry(filename)
    if not df.empty:
        st.dataframe(df[['Title', 'Text', 'Date']], use_container_width=True)