"""
Module for Implementation of Learning on the Border:
Active Learning in Imbalanced Data Classification.
"""

import random
import numpy as np
from sklearn import svm


class SSM:
    """
    SSM
    :params

    dataset: array-like
        input dataset without labels

    labels: list:
        labels of the input dataset
    task:str:
        learning task
    kernel:str
        kernel under consideration
    sample_size:int
        sample_size used. default is 59
    early_stopping:int
        specifies the stopping criteria when support vector stability is reached.


    """

    def __init__(self, dataset, labels, task, kernel,
                 sample_size=59, warm_start=4, early_stopping=2):
        self.dataset = dataset
        self.labels = labels
        self.kernel = kernel
        self.task = task
        self.sample_size = sample_size
        self.early_stopping = early_stopping
        self.warm_start = warm_start

        # try:
        #     if not isinstance(self.labels, np.ndarray):
        #         self.labels = np.array(self.labels)
        # except KeyError:
        #     raise 'wrong type'
        #
        # if not isinstance(self.dataset, np.ndarray):
        #     self.dataset = np.array(self.dataset)

        if not isinstance(task, str):
            raise TypeError('learning task must be a str specified as "binary" or "multiclass" ')

        if not isinstance(self.kernel, str):
            raise TypeError('Improper kernel initialisation passed')

        if not isinstance(self.warm_start, int):
            raise TypeError('warm_start initialisation must be an int')

        if self.warm_start < 4:
            raise ValueError('warm_start initialisation must be an int atleast 4 ')

        if not isinstance(self.early_stopping, int):
            raise TypeError('early_stopping initialisation must be an int')

        if self.early_stopping < 2:
            raise ValueError('early stopping initialisation must be atleast 2 ')

    def select_random_elements(self):
        sequence = range(len(self.dataset))
        random_indices = random.sample(sequence, self.sample_size)
        random_instances = self.dataset[random_indices]
        labels_rand_instances = self.labels[random_indices]
        return random_instances, labels_rand_instances

    def get_ssm(self):
        smm = svm.SVC(
            kernel=self.kernel,
            probability=False
        )
        return smm.fit(
            self.select_random_elements()[0],
            self.select_random_elements()[1]
        )

    def supervised_metric_svm_linear(self):

        smm = self.get_ssm()
        decision_boundary = smm.decision_function(
            self.select_random_elements()[0]
        )
        norm_weights = np.linalg.norm(smm.coef_)
        num_support_vectors = smm.n_support_
        distance = decision_boundary / norm_weights
        return distance, num_support_vectors

    def supervised_metric_svm(self):
        smm = self.get_ssm()
        relative_distance = smm.decision_function(
            self.select_random_elements()[0]
        )
        num_support_vectors = smm.n_support_
        return relative_distance, num_support_vectors

    def selection_metric(self, x):
        if self.kernel == 'linear':
            return self.supervised_metric_svm_linear()[x]
        return self.supervised_metric_svm()[x]

    def select_minimum_instance_binary(self, x):
        if self.task == 'binary':
            distance_space = self.selection_metric(x)
            minimum_distance_index = np.argmin(
                np.absolute(distance_space)
            )
            return minimum_distance_index
        return None

    def select_minimum_instance(self, x):
        if self.task == 'multiclass':
            relative_distance_space = np.absolute(self.selection_metric(x))
            max_relative_distance = [
                np.max(distance) for distance in relative_distance_space
            ]
            minimum_relative_distance_index = np.argmin(max_relative_distance)
            return minimum_relative_distance_index
        return None

    def get_active_instance_index(self, x):
        if self.task == 'binary':
            return self.select_minimum_instance_binary(x)
        return self.select_minimum_instance(x)

    def compute_support_vector_stability(self, support_vec_list):

        if self.task != 'binary' and \
                np.allclose(support_vec_list[-2], support_vec_list[-1]):
            return True
        if self.task == 'binary' and \
                len(np.unique(support_vec_list[-self.early_stopping:])) == 1:
            return True
        return None

    def compute_support_vector_ratio(self, support_vec, support_vec_list):
        if self.task == 'binary':
            return support_vec_list.append(
                np.round(np.absolute(
                    support_vec[0] / support_vec[-1]), 2)
            )
        return support_vec_list.append(support_vec)

    def select_active_learned_instances(self):
        should_continue = True
        selected_instances, support_vector_list, count = [], [], -1
        while should_continue:
            count += 1
            selected_instances.append(self.get_active_instance_index(0))
            self.compute_support_vector_ratio(self.selection_metric(1), support_vector_list)
            if self.warm_start < count and \
                    self.compute_support_vector_stability(support_vector_list):
                should_continue = False
        return selected_instances


if __name__ == '__main__':
    print('please import to use')
