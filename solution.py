import numpy
import csv
import pandas as pd

# functions to compute information gain:
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    if n_cat1 == 0 or n_cat2 == 0:
        return 0
    else:
        p1 = n_cat1 / (n_cat1 + n_cat2)
        p2 = n_cat2 / (n_cat1 + n_cat2)
        return - p1 * numpy.log2(p1) - p2 * numpy.log2(p2)

def get_min_feature_value(features: list, feature_index: int):
    min_value = features[0][feature_index]

    l = len(features)
    for i in range(l):
        if features[i][feature_index] < min_value:
            min_value = features[i][feature_index]
            
    return min_value

def get_max_feature_value(features: list, feature_index: int):
    max_value = features[0][feature_index]

    l = len(features)
    for i in range(l):
        if features[i][feature_index] >= max_value:
            max_value = features[i][feature_index]
            
    return max_value

def get_information_gain(features: list, labels: list, feature_index: int, feature_value: int):
    label1 = 0
    #number of labels in one subset of split:
    subset1_label1_count = 0
    subset1_label2_count = 0
    #number of labels in the other:
    subset2_label1_count = 0
    subset2_label2_count = 0

    n = len(features)
    for i in range(n):
        if features[i][feature_index] > feature_value:
            if labels[i] == label1:
                subset1_label1_count += 1
            else:
                subset1_label2_count += 1
        else:
            if labels[i] == label1:
                subset2_label1_count += 1
            else:
                subset2_label2_count += 1

    #initial number of labels in the set
    set_label1_count = subset1_label1_count + subset2_label1_count
    set_label2_count = subset1_label2_count + subset2_label2_count
    
    #size of each set
    e: int = subset1_label1_count + subset1_label2_count
    f: int = subset2_label1_count + subset2_label2_count
    l: int = set_label1_count + set_label2_count

    set_entropy: float = get_entropy(set_label1_count, 
                                     set_label2_count)
    
    subset1_entropy: float = e / l * get_entropy(subset1_label1_count, 
                                                 subset1_label2_count)
    
    subset2_entropy: float = f / l * get_entropy(subset2_label1_count,
                                                 subset2_label2_count)
    
    return (set_entropy - subset1_entropy - subset2_entropy)

def get_best_separation(features: list, labels: list) -> (int, int):
    max_gain = 0
    feature_index = 0
    feature_value = 0

    n = len(features[0])
    for i in range(n):
        min_f = get_min_feature_value(features, i)
        max_f = get_max_feature_value(features, i)
        for j in range(min_f, max_f):
             gain = get_information_gain(features, labels, i, j)
             if gain > max_gain:
                 max_gain = gain
                 feature_index = i
                 feature_value = j

    return feature_index, feature_value

# node of a decision tree:
class Node:
    def __init__(self, set):
        #set
        self.set = set

        #separation
        features, labels = self.process_set()
        sep_index, sep_value = get_best_separation(features, labels)
        self.sep_index = sep_index
        self.sep_value = sep_value

        #child nodes
        self.left_child = None
        self.right_child = None

    def process_set(self):
       features: list = []
       labels: list = []

       n = len(self.set)
       m = len(self.set[0])
       for i in range(n):
           feature: list = []
           for j in range(m - 1):
               feature.append(self.set[i][j])
           features.append(feature)    
           labels.append(self.set[i][m-1])

       return features, labels

    def is_pure_leaf(self) -> (bool, int):
        n = len(self.set)
        m = len(self.set[0])
        label = self.set[0][m - 1]
        for i in range(n):
            if self.set[i][m - 1] != label:
                return False, label
        return True, label 
    
# decision tree functionality:
def build(node: Node):
    if node.is_pure_leaf()[0]:
        return
    
    subset1: list = []
    subset2: list = []
    n = len(node.set)
    for i in range(n):
        if node.set[i][node.sep_index] > node.sep_value:
            subset1.append(node.set[i])
        else:
            subset2.append(node.set[i])
    
    right_child = Node(subset1)
    node.right_child = right_child
    build(right_child)

    left_child = Node(subset2)
    node.left_child = left_child
    build(left_child)

def make_decision(node: Node, feature: list) -> int:
    leaf, label = node.is_pure_leaf()
    if leaf:
        return label
    
    if feature[node.sep_index] > node.sep_value:
        return make_decision(node.right_child, feature)
    else:
        return make_decision(node.left_child, feature)

def read_csv_into_list(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:  
            l = len(row)
            for i in range(l):
                row[i] = int(row[i]) 
            data.append(row)
    return data

def write_list_into_csv(filepath, data: list):
    with open(filepath, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for value in data:
            csv_writer.writerow([value])

train_data = read_csv_into_list('train.csv')
root = Node(train_data)
build(root)

test_data = read_csv_into_list('test.csv')
result_data: list = []

n = len(test_data)
for i in range(n):
    label = make_decision(root, test_data[i])
    result_data.append(label)

write_list_into_csv('results.csv', result_data)

a = {"Abel": "buzi"}
a['Abel']

