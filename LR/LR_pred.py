import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

model_iv = pickle.load(open("model_iv.sav", 'rb'))
model_op = pickle.load(open("model_op.sav", 'rb'))

# call the below function to return the option price
def pred_option_LR(K, S, T):
    '''
    K = Strike price
    S = Stock price
    T = Time to expiry in years
    '''
    data = [[K,S,T]]
    df = pd.DataFrame(data, columns=["Strike", "Stock", "Time"])

    IV = float(model_iv.predict(df))

    data = [[K,S,IV,T]]
    df =  pd.DataFrame(data, columns=["Strike", "Stock", "IV", "Time"])

    option_price = float(model_op.predict(df))

    return option_price







