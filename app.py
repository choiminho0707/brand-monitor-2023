import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# 1. 페이지 설정 및 가독성 향상
st.set_page_config(page_title="2023 AI Brand Insights", layout="wide", initial_sidebar_state="expanded")

# 데이터 로드 함수 (일련번호 1번부터 시작)
def load_data(filename):
    if not os.path.exists(filename): 
        return pd.DataFrame()
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
        # 인덱스를 1부터 시작하도록 설정
        df.index = df.index + 1 
        return df
    except:
        return pd.DataFrame()

# 2. 사이드바 내비게이션
with st.sidebar:
    st.title("🚀 Navigate & Analysis")
    
    # 요청하신 노멀한 안내 문구 적용
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border: 1px solid #eee; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.9em; color: #444; line-height: 1.4;">
                🔎 Explore what's happening with your brand.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    menu = st.radio("Go to", ["📦 Product Insights", "💬 Testimonial Stories", "⭐ Review Analytics"], index=2)
    st.markdown("---")
    
    # 월 선택 슬라이더 (Review Analytics 전용)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sel_month = st.select_slider("Select Month", options=range(1, 13), format_func=lambda x: month_names[x-1])

# 3. 메인 콘텐츠 영역
if menu == "⭐ Review Analytics":
    st.title("⭐ Deep Learning Review Analysis")
    
    # [핵심] 503 에러 방지를 위해 로컬에서 분석 완료된 파일 로드
    df = load_data("reviews_analyzed.csv")
    
    if not df.empty:
        # 날짜 필터링
        df['Date_dt'] = pd.to_datetime(df['Date'])
        filtered = df[df['Date_dt'].dt.month == sel_month].copy()
        
        if not filtered.empty:
            # 성공 안내 문구는 요청에 따라 삭제했습니다.
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Sentiment Distribution")
                
                # 감성별 통계 계산
                chart_data = filtered.groupby('Sentiment')['Confidence'].agg(['count', 'mean']).reset_index()
                chart_data.columns = ['Sentiment', 'Review Count', 'Avg. Confidence']
                
                # [디자인] 요청에 따라 Avg. Confidence를 포함한 모든 라벨을 굵게(bold) 처리
                chart_data['Display Label'] = chart_data.apply(
                    lambda x: f"<b>{x['Sentiment']}</b><br><b>Avg. Confidence ({x['Avg. Confidence']:.4f})</b>", 
                    axis=1
                )
                
                # 시각화 차트 생성
                fig = px.bar(chart_data, 
                             x='Display Label', 
                             y='Review Count', 
                             color='Sentiment',
                             color_discrete_map={'POSITIVE': '#00b894', 'NEGATIVE': '#ff7675'})
                
                fig.update_traces(texttemplate='%{y}', textposition='outside')
                fig.update_layout(xaxis_title="", yaxis_title="Review Count", xaxis={'tickangle': 0})
                st.plotly_chart(fig, use_container_width=True)
                
            with c2:
                st.subheader("☁️ Word Cloud")
                text_content = " ".join(filtered['Text'])
                if text_content.strip():
                    wc = WordCloud(background_color="white", width=800, height=500).generate(text_content)
                    fig_wc, ax = plt.subplots(); ax.imshow(wc); ax.axis("off")
                    st.pyplot(fig_wc)
        else:
            st.warning(f"No reviews found for {month_names[sel_month-1]}.")
    else:
        st.error("reviews_analyzed.csv 파일을 찾을 수 없습니다. 로컬에서 먼저 분석 스크립트를 실행해 주세요.")

else: # Product Insights 또는 Testimonial Stories 메뉴
    st.title(f"{menu}")
    filename = "products.csv" if "Product" in menu else "testimonials.csv"
    df = load_data(filename)
    if not df.empty:
        st.dataframe(df, use_container_width=True)