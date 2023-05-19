db = {
    'user': 'root',
    'password': '1234',
    'host': 'localhost',
    'prot': 3006,
    'database': 'test'
}

DB_URL = f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['database']}?charset=utf8"