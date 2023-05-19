import pymysql
'''
# MySQL 연결 핑확인
def check_mysql_connection():
    try:
        # MySQL 연결 설정
        connection = pymysql.connect(
            host = 'localhost',
            user = 'root',
            password = '1234',
            database = 'MySQL'
        )
        
        # 연결 성공
        print("MySQL 연결 성공")

        # 연결 종료
        connection.close()
    except pymysql.Error as e:
        #연결 실패
        print("MySQL 연결 실패:", e)

# MySQL 연결 확인 실행
check_mysql_connection()
'''
def mysql_connection(data):
    # MySql 연결 설정
    connection = pymysql.connect(
        host = 'localhost',
        user = 'root',
        password = '1234',
        database = 'bitcoin'
    )

    # 커서 생성
    cursor = connection.cursor()

    # 데이터 삽입 쿼리
    sql = "INSERT INTO predictions (dateday, prediction) VALUES (%s, %s)"
    
    # 데이터 삽입 실행
    cursor.execute(sql, (data["value1"], data["value2"]))

    # 변경 사항 저장
    connection.commit()

    # 연결 종료
    cursor.close()
    connection.close()