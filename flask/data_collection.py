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
            file_path = "flask/data/bitcoinapi.csv"  # 파일 경로
            if os.path.isfile(file_path):
                # 기존 파일을 청크 단위로 읽어오기
                df_existing_chunks = pd.read_csv(file_path, chunksize=10000)
                df_existing_list = []
                for chunk in df_existing_chunks:
                    df_existing_list.append(chunk)
                df_existing = pd.concat(df_existing_list)
                df_existing.reset_index(drop=True, inplace=True)
                
                # 중복 제외하고 새로운 데이터 추가
                df_existing = pd.concat([df_existing, df])
            else:
                df_existing = df
            
            # 중복 제외한 데이터 저장
            df_existing.to_csv(file_path, mode='w', header=True, index=False)
            
            print(f"Bitcoin data from {data[0]['candle_date_time_kst']} is collected.")
            
            if data[-1]['candle_date_time_utc'][:10] == end_date:
                break
            
            start_date = data[-1]['candle_date_time_utc']
            url = f"https://api.upbit.com/v1/candles/minutes/1?market=KRW-BTC&to={start_date}&count=200"
        else:
            print("Error: Failed to collect bitcoin data.")
        
        time.sleep(1)

if __name__ == '__main__':
    start_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date = "2021-01-01T00:00:00Z"
    get_bitcoin_data(start_date, end_date)
