import requests
import time
import pandas as pd

MAX_DATA_COUNT = 30
CSV_FILE_PATH = "data.csv"
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
    # 데이터프레임으로 변환
    df = pd.DataFrame(data, columns=COLUMN_NAMES)

    # '0' 값을 필터링하여 저장
    filtered_df = df[df['opening_price'] != 0]

    # CSV 파일에 추가 저장
    filtered_df.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)
    #print("데이터 저장 완료")

def initialize_csv_file():
    # 첫 번째 줄에 열 이름 추가
    pd.DataFrame(columns=COLUMN_NAMES).to_csv(CSV_FILE_PATH, index=False)
    print("CSV 파일 초기화 완료")

def display_realtime_data():
    while True:
        data = upbit_api()

        # 데이터 처리 및 출력
        # 예시로서 간단하게 candle_date_time_kst와 trade_price 값을 출력합니다.
        #candle_date_time_kst = data[0]['candle_date_time_kst']
        #trade_price = data[0]['trade_price']
        #print(f"실시간: {candle_date_time_kst}")
        #print(f"실시간 가격: {trade_price}")

        # CSV 파일에 데이터 저장
        save_data_to_csv(data)

        # 최대 데이터 개수 확인 및 삭제
        check_and_delete_excess_data()

        time.sleep(5)  # 1분마다 업데이트

def check_and_delete_excess_data():
    # CSV 파일 읽기
    df = pd.read_csv(CSV_FILE_PATH)

    # 최대 데이터 개수 초과 여부 확인
    if len(df) > MAX_DATA_COUNT:
        # 가장 오래된 데이터 삭제
        df = df.tail(MAX_DATA_COUNT)

        # 수정된 데이터 다시 저장
        df.to_csv(CSV_FILE_PATH, index=False)

        print("최대 데이터 개수 초과로 인한 데이터 삭제")

# CSV 파일 초기화 (최초 실행시 한 번만 수행)

initialize_csv_file()
display_realtime_data()
