import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 데이터셋 파일의 경로
dataset_path = '../notebook/naver_bit_data.csv'

# 데이터 불러오기
data = pd.read_csv(dataset_path)

# 전처리: 특수 문자 제거 및 공백 제거
data['news_title'] = data['news_title'].apply(lambda x: re.sub(r"[^ㄱ-ㅣ가-힣a-zA-Z0-9\s]", "", x).strip())

# 결측치 제거
data.dropna(subset=['news_title'], inplace=True)

# 뉴스 제목과 레이블 추출
news_titles = data['news_title']
labels = data['label']  # 수정: 레이블 추출

# 데이터 분할
if len(news_titles) > 0:
    X_train, X_test, y_train, y_test = train_test_split(news_titles, labels, test_size=0.2, random_state=42)  # 수정: y_train과 y_test 설정
    
    # 텍스트 벡터화
    vectorizer = CountVectorizer()
    X_train_matrix = vectorizer.fit_transform(X_train)
    X_test_matrix = vectorizer.transform(X_test)

    # 모델 학습
    clf = MultinomialNB()
    clf.fit(X_train_matrix, y_train)

    # 예측 및 평가
    y_pred = clf.predict(X_test_matrix)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Accuracy: ", accuracy)
    print("Confusion Matrix: \n", cm)

    def predict_sentiment(text):
        text = re.sub(r"[^ㄱ-ㅣ가-힣a-zA-Z0-9\s]", "", text).strip()
        vectorized_text = vectorizer.transform([text])
        prediction = clf.predict(vectorized_text)[0]
        return "긍정" if prediction == 1 else "부정"

    # 데이터프레임에 레이블 열 추가
    data['label'] = pd.concat([labels, pd.Series(y_pred, index=y_test.index)], ignore_index=True)

    # csv 파일 다시 저장
    data.to_csv(dataset_path, index=False)
else:
    print("데이터가 없습니다.")
