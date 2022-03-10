# Roll 18EE35032
# Name Vibhanshu Ranjan
# Assignment Number 1

import numpy as np
import pandas as pd
from numpy import log2 as log
epis = np.finfo(float).eps


# It returns the entropy of the current table
def entropy_table(table):
    entropy = 0
    class_values = table['class'].unique()
    # print(values)
    for class_value in class_values:
        fraction = table['class'].value_counts()[class_value] / len(table['class'])
        entropy += -fraction * log(fraction)
    return entropy


# It returns the entropy of the given attribute
def entropy_attribute(table, attribute):
    class_values = table['class'].unique()
    attribute_values = table[attribute].unique()
    # print(class_values)
    # print(attribute_values)
    entropy_attr = 0
    for attribute_value in attribute_values:
        entropy_attr_val = 0
        den = len(table[attribute][table[attribute] == attribute_value])
        for class_value in class_values:
            num = len(table[attribute][table[attribute] == attribute_value][table['class'] == class_value])
            fraction = num / (den + epis)
            entropy_attr_val += -fraction * log(fraction + epis)
        fraction = den / len(table)
        entropy_attr += -fraction * entropy_attr_val
    return abs(entropy_attr)


# It returns the best information gain attribte
def best_attribute(table):
    IG = []
    # print(table.keys())
    for key in table.keys()[:-1]:
        IG.append(entropy_table(table) - entropy_attribute(table, key))
    return table.keys()[:-1][np.argmax(IG)]


# It returns the subtable
def sub_table(table, attribute, attribute_value):
    # print(table[table[attribute] == attribute_value])
    return (table[table[attribute] == attribute_value].reset_index(drop=True)).drop(attribute, axis=1)


# Creating decision tree as a dictionary
def create_tree(table, tree=None):
    att = best_attribute(table)
    # print(att)
    att_values = np.unique(table[att])
    table_temp = table
    if tree is None:
        tree = {}
        tree[att] = {}
    for att_value in att_values:
        subtable = sub_table(table_temp, att, att_value)
        clValue, counts = np.unique(subtable['class'], return_counts=True)
        # print(clValue)
        # print(counts)
        if len(counts) == 1:
            tree[att][att_value] = clValue[0]
        elif len(table.columns) == 2 :
            tree[att][att_value] = clValue[np.argmax(counts)]
        else:
            tree[att][att_value] = create_tree(subtable)
    return tree

# It prints the final decision tree
def print_tree(tree, d):
    for key1 in tree.keys():
        tree2 = tree[key1]
        for key2 in tree2.keys():
            for i in range(d):
                print('|   ', end='')
            if type(tree2[key2]) is str:
                print(key1, ' = ', key2, ': ', tree2[key2])
            else:
                print(key1, ' = ', key2)
                tree3 = tree2[key2]
                print_tree(tree3,d+1)




# It predicts the output
def Predict(data, tree):
    for key in tree.keys():
        value = data[key]
        if value not in tree[key]:
            return 'unacc'
        tree = tree[key][value]
        if type(tree) is dict:
            prediction = Predict(data, tree)
        else:
            prediction = tree
    return prediction


if __name__ == "__main__":

    col = ['price','maint','doors','persons','lug_boot','safety','class'] #Naming the columns of input data
    train_data = pd.read_csv('train.data',names=col,header=None)
    # print(train_data)

    Tree = create_tree(train_data) # create/build the decision tree
    print_tree(Tree, 0) # print the tree


    # Predicting for test data
    test_data = pd.read_csv('test.data',names=col,header=None)
    pred = [] # to store predicted values
    for i in range(len(test_data)):
        data_ = test_data.iloc[i]
        pred.append(Predict(data_, Tree))
        # print(pred[i])
    correct_pred = 0
    # Check score
    for i in range(len(pred)):
        # print(pred[i])
        if pred[i] == test_data['class'][i]:
            correct_pred += 1
    print('Prediction Accuracy (%) = ', (correct_pred * 100) / len(pred))







