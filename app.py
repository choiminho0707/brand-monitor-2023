import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# 1. 페이지 설정 및 가독성 최적화
st.set_page_config(page_title="2023 AI Brand Insights", layout="wide")

def load_data(filename):
    if not os.path.exists(filename): return pd.DataFrame()
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
        df.index = df.index + 1 
        return df
    except: return pd.DataFrame()

# 2. 사이드바 구성
with st.sidebar:
    st.title("🚀 Navigate & Analysis")
    st.markdown('<div style="background-color:#f8f9fa;padding:10px;border-radius:8px;border:1px solid #eee;margin-bottom:15px;">'
                '<p style="margin:0;font-size:0.85em;color:#444;">🔎 Brand Insights Dashboard</p></div>', unsafe_allow_html=True)
    
    menu = st.radio("Go to", ["📦 Product Insights", "💬 Testimonial Stories", "⭐ Review Analytics"], index=2)
    st.markdown("---")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sel_month = st.select_slider("Select Month", options=range(1, 13), format_func=lambda x: month_names[x-1])

# 3. 메인 콘텐츠 (Review Analytics)
if menu == "⭐ Review Analytics":
    # 제목 여백 최소화
    st.markdown('<h1 style="margin-top:-50px;">⭐ Deep Learning Review Analysis</h1>', unsafe_allow_html=True)
    
    df = load_data("reviews_analyzed.csv")
    
    if not df.empty:
        df['Date_dt'] = pd.to_datetime(df['Date'])
        filtered = df[df['Date_dt'].dt.month == sel_month].copy()
        
        if not filtered.empty:
            # 상단 레이아웃 (높이 압축)
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("📊 Sentiment Distribution")
                chart_data = filtered.groupby('Sentiment')['Confidence'].agg(['count', 'mean']).reset_index()
                chart_data['Display Label'] = chart_data.apply(
                    lambda x: f"<b>{x['Sentiment']}</b><br><b>Avg. Conf ({x['mean']:.4f})</b>", axis=1)
                
                # 차트 높이를 300으로 줄여 공간 확보
                fig = px.bar(chart_data, x='Display Label', y='count', color='Sentiment',
                             color_discrete_map={'POSITIVE': '#00b894', 'NEGATIVE': '#ff7675'},
                             height=300) 
                fig.update_layout(xaxis_title="", yaxis_title="Count", margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("☁️ Word Cloud")
                text_content = " ".join(filtered['Text'])
                if text_content.strip():
                    # 워드클라우드 크기 및 여백 조정
                    wc = WordCloud(background_color="white", width=800, height=400).generate(text_content)
                    fig_wc, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    st.pyplot(fig_wc)

            # [핵심] 구분선 및 데이터프레임 위치 상향 조정
            st.markdown('<hr style="margin:10px 0;">', unsafe_allow_html=True)
            st.subheader(f"📜 Original Reviews: {month_names[sel_month-1]} 2023")
            
            # 테이블 높이를 고정하여 차트와 함께 한 화면에 보이게 함
            st.dataframe(filtered[['Date', 'Text', 'Sentiment', 'Confidence']], 
                         use_container_width=True, height=250)
            
        else:
            st.warning(f"No reviews for {month_names[sel_month-1]}.")
    else:
        st.error("reviews_analyzed.csv missing.")

else: # Other menus
    st.title(menu)
    filename = "products.csv" if "Product" in menu else "testimonials.csv"
    df = load_data(filename)
    if not df.empty: st.dataframe(df, use_container_width=True)