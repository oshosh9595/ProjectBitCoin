# noise cosine generator ------------------------------------------------------------------
# 균일한 노이즈가 추가된 코사인파
import numpy as np

def fetch_cosine_values(seq_len, frequency=0.1, noise=0.1):
    np.random.seed(101) # 실험 복제
    x = np.arange(0.0, seq_len, 1.0)
    return np.cos(2 * np.pi * frequency * x) + np.random.uniform(low=noise, high=noise, size=seq_len)

# stock price to pickle -------------------------------------------------------------------
# Quandl API로 종목 데이터 가져오기
import os
import pickle
import quandl

# string to datetime format
def date_obj_to_str(date_obj):
    return date_obj.strftime('%Y-%m-%d')

# path에 pickle 저장
def save_pickle(something, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as fh:
        pickle.dump(something, fh, pickle.DEFAULT_PROTOCOL)

# load pickle, return object
def load_pickle(path):
    with open(path, 'rb') as fh:
      return pickle.load(fh)

def fetch_stock_price(symbol,
                      from_date,
                      to_date,
                      cache_path="./tmp/prices/"):
    assert(from_date <= to_date)
    filename = "{}_{}_{}.pk".format(symbol, str(from_date), str(to_date))
    price_filepath = os.path.join(cache_path, filename)
    try:
        prices = load_pickle(price_filepath)
        print("loaded from", price_filepath)
    except IOError:
        historic = quandl.get("WIKI/" + symbol,
                              start_date = date_obj_to_str(from_date),
                              end_date = date_obj_to_str(to_date))
        prices = historic["Adj. Close"].tolist()
        save_pickle(prices, price_filepath)
    return prices

# dataset formater -------------------------------------------------------------------------
# sliding window
def format_dataset(values, temporal_features):
    feat_splits = [values[i:i + temporal_features] for i in range(len(values) - temporal_features)]
    feats = np.vstack(feat_splits)
    labels = np.array(values[temporal_features:])
    return feats, labels

# helper -----------------------------------------------------------------------------------
# matrix to 1D array
def matrix_to_array(m):
    return np.asarray(m).reshape(-1)