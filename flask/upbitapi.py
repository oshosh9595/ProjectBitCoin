import requests
import time
import pandas as pd

MAX_DATA_COUNT = 30
CSV_FILE_PATH = "../data/data.csv"
COLUMN_NAMES = ['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_price', 'candle_acc_trade_volume']

def upbit_api():
    url = "https://api.upbit.com/v1/candles/minutes/1"
    params = {
        "market": "KRW-BTC",
        "count": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data

def save_data_to_csv(data):
    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    filtered_df = df[df['opening_price'] != 0]
    filtered_df.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)

def collect_realtime_data():
    while True:
        data = upbit_api()
        save_data_to_csv(data)
        check_and_delete_excess_data()
        time.sleep(1)

def check_and_delete_excess_data():
    df = pd.read_csv(CSV_FILE_PATH)
    if len(df) > MAX_DATA_COUNT:
        df = df.tail(MAX_DATA_COUNT)
        df.to_csv(CSV_FILE_PATH, index=False)
        print("최대 데이터 개수 초과로 인한 데이터 삭제")

def initialize_csv_file():
    pd.DataFrame(columns=COLUMN_NAMES).to_csv(CSV_FILE_PATH, index=False)
    print("CSV 파일 초기화 완료")

# CSV 파일 초기화 (최초 실행시 한 번만 수행)
initialize_csv_file()
