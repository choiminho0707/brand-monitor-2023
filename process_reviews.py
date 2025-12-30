import pandas as pd
from transformers import pipeline
import os

def run_analysis():
    # 1. 파일 존재 확인
    if not os.path.exists("reviews.csv"):
        print("❌ 에러: reviews.csv 파일이 같은 폴더에 없습니다!")
        return

    # 2. 고성능 모델 로드 (로컬은 메모리가 충분하므로 distilbert 사용)
    print("🔄 AI 모델을 불러오는 중입니다... (잠시만 기다려 주세요)")
    analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # 3. 데이터 읽기
    df = pd.read_csv("reviews.csv", encoding='utf-8-sig')
    print(f"📊 총 {len(df)}개의 리뷰를 분석 시작합니다.")

    # 4. 감성 분석 수행
    texts = df['Text'].tolist()
    results = analyzer(texts)

    # 5. 결과 저장 (라벨을 POSITIVE/NEGATIVE로 변환)
    df['Sentiment'] = [r['label'] for r in results]
    df['Confidence'] = [r['score'] for r in results]

    # 6. 최종 파일 저장
    df.to_csv("reviews_analyzed.csv", index=False, encoding='utf-8-sig')
    print("✅ 분석 완료! 'reviews_analyzed.csv' 파일이 생성되었습니다.")

if __name__ == "__main__":
    run_analysis()