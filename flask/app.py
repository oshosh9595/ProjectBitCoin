from flask import Flask, render_template, request, jsonify
import schedule
import time
import logging
import pandas as pd
import threading
from model import model_prediction
from database import mysql_connection
from upbitapi import collect_realtime_data

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

@app.route('/')
def job():
    logging.info("데이터 수집 시작")
    collect_realtime_data()

def collect_and_predict():
    logging.info("collect_and_predict 실행")
    # 데이터 로드
    data = pd.read_csv("data.csv")
    # 모델에 데이터 삽입
    y_pred_dict = model_prediction(data)
    # 데이터베이스에 삽입
    mysql_connection(y_pred_dict)

def schedule_job():
    schedule.every(1).minutes.do(collect_and_predict).tag('collect_and_predict') 
    # 3분마다 collect_and_predict 실행 tag는 안됨
    while True:
        schedule.run_pending() # 스케줄링된 작업 실행
        time.sleep(1) # 1초 대기

schedule_thread = threading.Thread(target=schedule_job) # schedule_job 쓰레드
schedule_thread.start()
if __name__ == "__main__":
    app.run(debug=False, threaded=True)
