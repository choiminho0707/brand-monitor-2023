import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import re # 정규표현식 모듈 추가

def scrape_reviews_only():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    base_urls = {
        "Products": "https://web-scraping.dev/products",
        "Testimonials": "https://web-scraping.dev/testimonials",
        "Reviews": "https://web-scraping.dev/reviews"
    }
    
    all_raw_data = []
    
    for category, url in base_urls.items():
        print(f"🚀 {category} 섹션에서 분석 소스 수집 중...")
        for page in range(1, 11): 
            try:
                res = requests.get(f"{url}?page={page}", headers=headers, timeout=10)
                if res.status_code != 200: break
                
                soup = BeautifulSoup(res.text, 'html.parser')
                items = soup.select('.review, .testimonial, .product, .card-body, p, blockquote')
                
                page_count = 0
                for item in items:
                    text = item.get_text(separator=' ', strip=True)
                    
                    # [핵심 수정] 문장 끝에 붙은 가격(예: 24.99) 제거 정규표현식
                    # \s? : 공백이 있을수도 없을수도 있음
                    # \d+ : 숫자 하나 이상
                    # \. : 마침표
                    # \d{2} : 숫자 정확히 2자리
                    # $ : 문장의 끝을 의미
                    text = re.sub(r'\s?\d+\.\d{2}$', '', text)
                    
                    if len(text) > 30: 
                        all_raw_data.append(text)
                        page_count += 1
                
                if page_count == 0: break 
                time.sleep(0.05)
            except: break

    unique_texts = list(set(all_raw_data))
    final_rows = []
    
    for i, text in enumerate(unique_texts):
        month = (i % 12) + 1
        day = random.randint(1, 28)
        
        final_rows.append({
            "Category": "Reviews",
            "Title": f"Customer Feedback #{i+1}",
            "Text": text,
            "Date": f"2023-{month:02d}-{day:02d}"
        })

    df = pd.DataFrame(final_rows)
    df.to_csv("reviews.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 가격 정보가 제거된 {len(df)}개의 데이터가 'reviews.csv'로 저장되었습니다.")

if __name__ == "__main__":
    scrape_reviews_only()