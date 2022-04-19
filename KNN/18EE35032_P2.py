# Roll 18EE35032
# Name Vibhanshu Ranjan
# Assignment Number 2

import pandas as pd
import numpy as np

def KNN(data,train_data,K):
    dist = [];  # will contain euclidean distance of test data with each training data and target class of the training data
    train_data_ = train_data.iloc[:,:-1];
    for i in range(len(train_data_)):
        tdata = train_data_.iloc[i];
        euclidean_dist = np.linalg.norm(data-tdata); # finding euclidean distance
        dist.append([euclidean_dist,train_data['target'][i]]);
    print(dist)
    dist = sorted(dist,key=lambda x:x[0]) # sorting euclidean distances
    # print(dist)
    pos = 0;
    for i in range(K): # finding majority classes
        if dist[i][1]==1:
            pos = pos+1;
    return 1 if (pos > K-pos) else 0;


if __name__ == "__main__":
    train_data = pd.read_csv('project2.csv')
    # print(train_data)
    test_data = pd.read_csv('project2_test.csv');
    pred = []
    for i in range(len(test_data)):
        data = test_data.iloc[i]
        K = 16;
        pred.append(KNN(data,train_data,K)) # predicting class for each test data
    print(pred)
    f = open("18EE35032_P2.out.txt", "a")
    for i in range(len(pred)):
        f.write(str(pred[i])+" ")
    f.close()