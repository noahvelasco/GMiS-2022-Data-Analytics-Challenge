
import numpy as np
import pandas as pd
import math 
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error 


def split_train_test(X,y,percent_train=0.9,seed=None):
    if seed!=None:
        np.random.seed(seed)
    ind = np.random.permutation(X.shape[0])
    train = ind[:int(X.shape[0]*percent_train)]
    test = ind[int(X.shape[0]*percent_train):]
    return X[train],X[test], y[train], y[test] 



def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/y_true.shape[0]


if __name__ == "__main__":
   
    #Noahs Data Path
    data_path = 'C:\\Users\\Noah\\Desktop\\cahsi_data_2022\\'  # Use your own path her

    df_d1 = pd.read_csv(data_path+'D1.csv')
    df_d2 = pd.read_csv(data_path+'D2.csv')
    
    
    #Train
    X = df_d1[:].values
    X_d1 = X[:,:-1]
    y_d1 = X[:,-1]
    
    X_train, X_test, y_train, y_test = split_train_test(X_d1,y_d1,seed=20)

    
    model = MLPClassifier(activation= 'relu',solver='adam', alpha=1e-5, batch_size = 100 ,learning_rate='adaptive',momentum=0.95,  hidden_layer_sizes=(200), verbose=True, random_state=1)
   
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    print('Training iterations  {} '.format(model.n_iter_))  
    
    start = time.time()       
    pred = model.predict(X_test)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))   
    print('Accuracy: {0:.6f}'.format(accuracy(y_test,pred)))
    
    print("MSE",mean_squared_error(y_test, pred))
    
    
    #export
    with open("answers.txt", "w") as txt_file:
        for line in pred:
            txt_file.write(str(line) + "\n") # works with any number of elements in a line
    
    
    
    
    
    
    
    
