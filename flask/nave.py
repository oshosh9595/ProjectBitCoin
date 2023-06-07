import requests
from bs4 import BeautifulSoup
import pandas as pd
import pyupbit
import logging

df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=200)
df = df.dropna()
dates = df.index
df['Date'] = dates
df = df.set_index('Date')

df['Price'] = 0
for i in range(0, 30):
    if df['close'][i] < df['close'][i+1]:
        df.loc[df.index[i], 'Price'] = 1
    else:
        df.loc[df.index[i], 'Price'] = 0

df.to_csv('../data/upbit_data.csv')
logging.info(df)

price_data = pd.read_csv('../data/upbit_data.csv')

df_0 = price_data[price_data['Price'] == 0]['Date']
date_0 = [str(date)[:10].replace('-', '') for date in df_0]
logging.info(date_0)

df_1 = price_data[price_data['Price'] == 1]['Date']
date_1 = [str(date)[:10].replace('-', '') for date in df_1]
logging.info(date_1)

def get_news_content(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    content = soup.select_one('#articleBodyContents').get_text(separator=' ', strip=True)
    return content

def naver_bitcoin_news(dates):
    error_cnt = 0
    base_url = 'https://search.naver.com/search.naver?where=news&query=비트코인&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={}&de={}&mynews=1&office_type=1&office_section_code=1&news_office_checked=1001&nso=so:r,p:from{}to{}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
    }

    result_list = []

    for date in dates:
        start_date = date
        end_date = date
        url = base_url.format(start_date, end_date, start_date, end_date)
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            news_list = soup.select('ul.list_news li')
            for news in news_list:
                try:
                    news_title = news.select_one('a.news_tit').text.strip()
                    news_url = news.select_one('a.news_tit')['href']
                    news_content = get_news_content(news_url)
                    result_list.append([news_title, news_content])
                except:
                    error_cnt += 1
        logging.info(f"Error count: {error_cnt}")

    return result_list

result_list_0 = naver_bitcoin_news(date_0)
title_df_0 = pd.DataFrame(result_list_0, columns=['news_title', 'news_content'])
title_df_0['stock_price_fluctuation'] = 0
logging.info(title_df_0)

result_list_1 = naver_bitcoin_news(date_1)
title_df_1 = pd.DataFrame(result_list_1, columns=['news_title', 'news_content'])
title_df_1['stock_price_fluctuation'] = 1
logging.info(title_df_1)

title_df = pd.concat([title_df_0, title_df_1])
title_df.to_csv('../data/naver_bit_data.csv', index=False, encoding='utf-8')
logging.info(title_df)
