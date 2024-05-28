import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np 
from copy import deepcopy
from scipy.spatial import distance

from collections import Counter

TEST_SIZE = 0.4


#------------------------------------
#KD tree 
def partition_sort(arr, k, axis):
    """
    param 
        arr: input 
        k: splitting idx 
        axis: splitting axis
    """
    key_array = arr[:,axis] 
    partition_idx = np.argpartition(key_array, k)
    return arr[partition_idx]
    

def max_heap_replace(heap, new_node, key=lambda x: x[1]):
    """
    max heap add 
    param 
        heap
        new_node
    """
    heap[0] = new_node
    root, child = 0, 1
    end = len(heap) - 1
    while child <= end:
        if child < end and key(heap[child]) < key(heap[child + 1]):
            child += 1
        if key(heap[child]) <= key(new_node):
            break
        heap[root] = heap[child]
        root, child = child, 2 * child + 1
    heap[root] = new_node


def max_heap_push(heap, new_node, key=lambda x: x[1]):
    """
    max heap add 
    param 
        heap
        new_node
    """
    heap.append(new_node)
    pos = len(heap) - 1
    while 0 < pos:
        parent_pos = pos - 1 >> 1 # half the pos
        if key(new_node) <= key(heap[parent_pos]):
            break
        heap[pos] = heap[parent_pos]
        pos = parent_pos
    heap[pos] = new_node


class KDNode(object):
    def __init__(self, data=None, label=None, left=None, right=None, axis=None, parent=None):
        """
        param 
            data
            label
            left
            right
            axis: splitting axis
            parent: parent node 
        """
        self.data = data
        self.label = label
        self.left = left
        self.right = right
        self.axis = axis
        self.parent = parent


class KDTree(object):
    def __init__(self, X, y=None):
        """
        param: X,y
        """
        self.root = None
        self.y_valid = False if y is None else True
        self.create(X, y)

    def create(self, X, y=None):
        """
        param: X,y
        return: KDNode
        """

        def create_(X, axis, parent=None):
            """
            param:
                X
                axis: splitting axis
                parent: parent node 
            return: KDNode
            """
            n_samples = np.shape(X)[0]
            if n_samples == 0:
                return None
            mid = n_samples >> 1 # half the pos
            X = partition_sort(X, mid, axis)

            if self.y_valid:
                kd_node = KDNode(X[mid][:-1], X[mid][-1], axis=axis, parent=parent)
            else:
                kd_node = KDNode(X[mid], axis=axis, parent=parent)

            next_axis = (axis + 1) % k_dimensions
            kd_node.left = create_(X[:mid], next_axis, kd_node)
            kd_node.right = create_(X[mid + 1:], next_axis, kd_node)
            return kd_node

        print('building kd-tree...')
        k_dimensions = np.shape(X)[1]
        if y is not None:
            X = np.hstack((np.array(X), np.array([y]).T))
        self.root = create_(X, 0)

    def search_knn(self, point, k, p = 2 ):
        """
        param:
            point: samples
            k: neighbor
            dist: distance
        """

        def search_knn_(kd_node):
            """
            param
                kd_node: KDNode
            """
            if kd_node is None:
                return
            data = kd_node.data
            distance = fun_dist(data)
            if len(heap) < k:
                #add new 
                max_heap_push(heap, (kd_node, distance))
            elif distance < heap[0][1]:
                #replace top
                max_heap_replace(heap, (kd_node, distance))

            axis = kd_node.axis
            if abs(point[axis] - data[axis]) < heap[0][1] or len(heap) < k:
                #intersection with mini sphere  OR element number in heap is less than k
                search_knn_(kd_node.left)
                search_knn_(kd_node.right)
            elif point[axis] < data[axis]:
                search_knn_(kd_node.left)
            else:
                search_knn_(kd_node.right)

        if self.root is None:
            raise Exception('kd-tree must be not null.')
        if k < 1:
            raise ValueError("k must be greater than 0.")
        
        fun_dist = lambda x: distance.minkowski(x, point, p)

        heap = []
        search_knn_(self.root)
        return sorted(heap, key=lambda x: x[1])

    def query(self, point,k=1, p=2):
        """
        param 
            point: query target
            p: order of the norm
        """
        search_res = self.search_knn(point, k, p)
        xx = [x[0].data for x in search_res[:k] ]
        yy = [x[0].label for x in search_res[:k]]
        return xx,yy 
    
#------------------------------------
#KNeighborsClassifier
class KNeighborsClassifier():
    """
    fit:
        .fit(evidence, labels)
    predict:
        .predict(X_test)
    
    """
    def __init__(self,n_neighbors=1,p=2):
        """
        param:
            n_neighbors
            p: Minkowski p-norm
        """
        self.k = n_neighbors
        self.p = p 
        
        self.tree = None 

    def fit(self,X, y ):
        """
        param:X,y : list -> ndarray
        """      
        self.tree = KDTree( np.array(X) , np.array(y) )
        
    def predict(self, testX ):
        """
        param:
            testX : ndarray
        return:
            pred 
        """
        pred_ = [] 
        
        testX = np.array( testX ) 
        for i in range(len(testX)):
            xx,yy = self.tree.query(testX[i,:], k = self.k,  p=self.p )
            if type( yy ) == list:
                count_ = Counter( yy )
                pred_.append( count_.most_common(1)[0][0] ) 
            else:
                pred_.append(  yy )
        
        assert len(pred_) == len(testX)
        
        return np.array(pred_ )




def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    
    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity,f1 = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"F1 Score: {100 * f1:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    evidence = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = float(0)
    specificity = float(0)

    total_positive = float(0)
    total_negative = float(0)
    
    precision = float(0)
    pred_pos = float(0)
    
    for label, prediction in zip(labels, predictions):

        if label == 1:
            total_positive += 1
            if label == prediction:
                sensitivity += 1

        if label == 0:
            total_negative += 1
            
            if label == prediction:
                specificity += 1
                
        if prediction == 1:
            pred_pos += 1
            if label == prediction:
                precision += 1
            

    sensitivity /= total_positive
    specificity /= total_negative
    
    #F1 measure.
    recall = sensitivity
    precision /= pred_pos
    f1 = 2*precision*recall/(precision + recall)
    
    return sensitivity, specificity, f1



if __name__ == "__main__":
    main()
