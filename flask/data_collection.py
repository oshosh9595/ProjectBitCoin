import requests
import json
import time
from datetime import datetime
import pandas as pd
import os

def get_bitcoin_data(start_date, end_date):
    url = f"https://api.upbit.com/v1/candles/minutes/5?market=KRW-BTC&to={start_date}&count=200"
    headers = {"accept": "application/json"}
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = json.loads(response.text)
            if len(data) == 0:
                print("No data is available.")
                break
            df = pd.DataFrame(data, columns=['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_price', 'candle_acc_trade_volume'])
            df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'])
            df = df.drop_duplicates(subset='candle_date_time_kst')
            df = df.sort_values('candle_date_time_kst')
            
            # 기존 파일이 존재하는지 확인
            if os.path.isfile("../data/bitcoin2023_t.csv"):
                df_existing = pd.read_csv("../data/bitcoin2023_t.csv")
                df = pd.concat([df_existing, df])
            
            df.to_csv("../data/bitcoin2023_t.csv", mode='w', header=True, index=False)
            
            print(f"Bitcoin data from {data[0]['candle_date_time_kst']} is collected.")
            
            if data[-1]['candle_date_time_utc'][:10] == end_date:
                break
            
            start_date = data[-1]['candle_date_time_utc']
            url = f"https://api.upbit.com/v1/candles/minutes/5?market=KRW-BTC&to={start_date}&count=200"
        else:
            print("Error: Failed to collect bitcoin data.")
        
        time.sleep(1)

if __name__ == '__main__':
    start_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date = "2023-01-01T00:00:00Z"
    get_bitcoin_data(start_date, end_date)
