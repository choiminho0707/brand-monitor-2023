import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re, os, gc

# 페이지 설정
st.set_page_config(page_title="2023 AI Brand Insights", layout="wide")

# [해결 핵심] 메모리 부족을 방지하는 스마트 로딩 함수
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    is_render = "RENDER" in os.environ
    try:
        # Render 서버면 가장 용량이 작은 모델을 사용해 503 에러 방지
        model_name = "prajjwal1/bert-tiny" if is_render else "distilbert-base-uncased-finetuned-sst-2-english"
        return pipeline("sentiment-analysis", model=model_name, device=-1)
    except:
        return None

def load_data(filename):
    if not os.path.exists(filename): return pd.DataFrame()
    df = pd.read_csv(filename, encoding='utf-8-sig')
    # 일련번호 1번 시작
    df.index = df.index + 1
    return df

# 사이드바 구성 및 요청 문구 반영
with st.sidebar:
    st.title("🚀 Navigate & Analysis")
    st.markdown(f'<div style="background-color:#f8f9fa;padding:12px;border-radius:8px;border:1;margin-bottom:20px;">'
                f'🔎 Explore what\'s happening with your brand.</div>', unsafe_allow_html=True)
    
    menu = st.radio("Go to", ["📦 Product Insights", "💬 Testimonial Stories", "⭐ Review Analytics"], index=2)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sel_month = st.select_slider("Select Month", options=range(1, 13), format_func=lambda x: month_names[x-1])

if menu == "⭐ Review Analytics":
    st.title("⭐ Deep Learning Review Analysis")
    df = load_data("reviews.csv")
    if not df.empty:
        df['Date_dt'] = pd.to_datetime(df['Date'])
        filtered = df[df['Date_dt'].dt.month == sel_month].copy()
        
        if not filtered.empty:
            # [해결] 스피너 메시지 최적화 및 분석 실행
            with st.status(f"Analyzing {month_names[sel_month-1]} reviews...", expanded=True) as status:
                st.write("Loading AI model into memory...")
                analyzer = load_sentiment_model()
                if analyzer:
                    st.write("Computing sentiment scores...")
                    results = analyzer(filtered['Text'].tolist())
                    filtered['Sentiment'] = ["POSITIVE" if r['label'] in ['LABEL_1', 'POSITIVE'] else "NEGATIVE" for r in results]
                    filtered['Confidence'] = [r['score'] for r in results]
                    # 분석 완료 후 즉시 모델 관련 메모리 수동 해제
                    del results
                    gc.collect()
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

            # 시각화 부분
            c1, c2 = st.columns(2)
            with c1:
                chart_data = filtered.groupby('Sentiment')['Confidence'].agg(['count', 'mean']).reset_index()
                chart_data['Display Label'] = chart_data.apply(
                    lambda x: f"<b>{x['Sentiment']}</b><br>Avg. Confidence ({x['mean']:.4f})", axis=1)
                fig = px.bar(chart_data, x='Display Label', y='count', color='Sentiment',
                             color_discrete_map={'POSITIVE': '#00b894', 'NEGATIVE': '#ff7675'})
                fig.update_layout(xaxis_title="", yaxis_title="Review Count")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                wc = WordCloud(background_color="white").generate(" ".join(filtered['Text']))
                fig_wc, ax = plt.subplots(); ax.imshow(wc); ax.axis("off")
                st.pyplot(fig_wc)
else:
    st.title(menu)
    df = load_data("products.csv" if "Product" in menu else "testimonials.csv")
    st.dataframe(df, use_container_width=True)