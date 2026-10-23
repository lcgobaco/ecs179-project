'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    # the stage 2 dataset is pre-partitioned into a training file and a testing file
    dataset_train_file_name = 'train.csv'
    dataset_test_file_name = 'test.csv'
    # scale the pixel features from [0, 255] into [0, 1]
    normalize = True

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_file(self, file_name):
        # each line: label,feature1,feature2,...,feature784 (all integers)
        raw = np.loadtxt(self.dataset_source_folder_path + file_name, delimiter=',', dtype=np.float32)
        X = raw[:, 1:]
        y = raw[:, 0].astype(np.int64)
        if self.normalize:
            X = X / 255.0
        return {'X': X, 'y': y}

    def load(self):
        # reuse the parsed data when running several experiments on the same loader
        if self.data is not None:
            return self.data
        print('loading data...')
        train_data = self.load_file(self.dataset_train_file_name)
        test_data = self.load_file(self.dataset_test_file_name)
        print('train:', train_data['X'].shape, 'test:', test_data['X'].shape)
        self.data = {'train': train_data, 'test': test_data}
        return self.data
