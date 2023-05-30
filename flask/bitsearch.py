import requests
from bs4 import BeautifulSoup
import csv
import signal

url = "https://search.naver.com/search.naver?query=비트코인&where=news"
articles = []

def start_one():
    start = 1  # 시작 페이지 번호
    while True:
        page_url = f"{url}&start={start}"
        response = requests.get(page_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_area = soup.find_all('div', class_='news_area')

        if not news_area:  # news_area가 비어있으면 마지막 페이지라고 가정
            break

        articles.extend(news_area)
        start += 10  # 다음 페이지로 이동

        # 사용자 입력 감지
        if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
            print("사용자에 의해 프로그램이 중단되었습니다.")
            break
    
    if articles:
        with open('./data/news.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['뉴스 제목', '뉴스 내용', '뉴스 날짜'])
            for article in articles:
                news_title_elem = article.find('a', class_='news_tit')
                news_title = news_title_elem.text.strip() if news_title_elem else ""

                news_content_elem = article.find('a', class_='api_txt_lines dsc_txt_wrap')
                news_content = news_content_elem.text.strip() if news_content_elem else ""

                news_date_elem = article.find('span', class_='info')
                news_date = news_date_elem.text.strip() if news_date_elem else ""

                writer.writerow([news_title, news_content, news_date])

        print("뉴스 제목, 내용, 날짜가 성공적으로 저장되었습니다.")
    else:
        print("수집된 뉴스가 없습니다.")

# KeyboardInterrupt 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal.default_int_handler)

start_one()
