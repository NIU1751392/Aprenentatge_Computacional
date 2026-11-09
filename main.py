from data import load_data
from preprocessing import *

def main():
    df = load_data()

    X,y = prepare_data(df) 

    X_train, X_test, y_train, y_test = split_data(X,y)

    #print("Train:", X_train.shape)
    #print("Test:", X_test.shape)

if __name__=="__main__":
    main()

