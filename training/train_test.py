
import numpy as np
import pandas as pd

# functions to test are imported from train.py
from train import train_test_split_preprocess, train_model, get_model_metrics

"""A set of simple unit tests for protecting against regressions in train.py"""

import unittest
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import os 


class TestTrainTestSplitPreprocess(unittest.TestCase):

    def setUp(self):
        # create sample data
        train_features = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        train_label = pd.DataFrame({'label': [0, 1, 0]})
        test_features = pd.DataFrame({'feature1': [4, 5, 6], 'feature2': [7, 8, 9]})
        test_label = pd.DataFrame({'label': [1, 0, 1]})
        train_features.to_pickle('./trainfeatures.pkl')
        train_label.to_pickle('./trainlabel.pkl')
        test_features.to_pickle('./testfeatures.pkl')
        test_label.to_pickle('./testlabel.pkl')

    def tearDown(self):
        # remove sample data
        for file in ['trainfeatures.pkl', 'trainlabel.pkl', 'testfeatures.pkl', 'testlabel.pkl']:
            os.remove(f"./{file}")

    def test_train_test_split_preprocess(self):
        dataset = DummyDataset('./')
        result = train_test_split_preprocess(dataset)

        # check that X_train, y_train, X_test, y_test are of expected shape
        self.assertEqual(result[0].shape, (3, 2, 1))
        self.assertEqual(result[1].shape, (3,))
        self.assertEqual(result[2].shape, (3, 2, 1))
        self.assertEqual(result[3].shape, (3,))

        # check that label encoder and one-hot encoding were applied correctly
        lb = LabelEncoder()
        lb.fit(['0', '1'])
        expected_y_train = np_utils.to_categorical(lb.transform(['0', '1', '0']))
        expected_y_test = np_utils.to_categorical(lb.transform(['1', '0', '1']))
        np.testing.assert_array_equal(result[1], expected_y_train)
        np.testing.assert_array_equal(result[3], expected_y_test)  
    
class DummyDataset:
     def __init__(self, path):
        self.path = path

     def to_path(self):
        return self.path

if __name__ == '__main__':
    unittest.main()







