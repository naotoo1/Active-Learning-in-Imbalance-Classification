"""
Implementation of Active learning for learning vector quantization
"""

import random
import torch
import numpy as np
import prototorch as pt
import prototorch.models
import pytorch_lightning as pl
import torch.utils.data


class SSM:
    """
    SSM

    dataset:
        input dataset without labels
    label: array-like:
        labels of the input dataset
    sample_size:int:
        random sample from the pool for each iteration
    warm_start: int:
        Initial number of iteration(s) needed before prototype stability is reached
    early_stopping: int:
        A small value used for the stopping criteria when prototype stability is reached.
    max_epochs: int:
        max_epochs used by the supervised metric used for the active learning
    ord: str:
        Order of the norm : parameter for matrix magnitude computation
    batch_size: int:
        batch_size used by the supervised  active learning metric
    close_to_boundary: bool:
        True for border learning and False otherwise.
    top_n: bool:
        True for the closest and False amongst top p%

    """

    def __init__(self, dataset, labels, sample_size, warm_start,
                 early_stopping, max_epochs, ord, batch_size=10, top_n=True,
                 close_to_boundary=True):
        self.dataset = dataset
        self.labels = labels
        self.sample_size = sample_size
        self.early_stopping = early_stopping
        self.warm_start = warm_start
        self.ord = ord
        self.close_to_boundary = close_to_boundary
        self.top_n = top_n
        self.max_epochs = max_epochs
        self.top_p = 5
        self.batch_size = batch_size

    @staticmethod
    def reset_weights(m):
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    def select_random_elements(self):

        sequence = range(len(self.dataset))
        random_indices = random.sample(sequence, self.sample_size)
        random_instances = self.dataset[random_indices]
        labels_rand_instances = self.labels[random_indices]
        return random_instances, labels_rand_instances

    def get_data_loader(self):

        random_instances, labels_random_instances = self.select_random_elements()
        train_ds = torch.utils.data.TensorDataset(
            random_instances,
            labels_random_instances
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            self.batch_size
        )
        return train_loader, random_instances, train_ds

    def supervised_metric(self):

        load_data, random_instances, train_ds = self.get_data_loader()
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            enable_progress_bar=False,
            enable_model_summary=False,
            gpus=1
        )
        model = pt.models.glvq.GMLVQ(
            hparams=dict(
                input_dim=2,
                latent_dim=2,
                distribution=[1, 1],
                proto_lr=0.01,
                bb_lr=0.01
            ),
            prototypes_initializer=pt.initializers.SMCI(
                train_ds,
                noise=0.1
            ), optimizer=torch.optim.Adam
        )

        model.apply(self.reset_weights)
        trainer.fit(model, load_data)
        relative_distance = model.compute_distances(random_instances)
        learned_components = model.prototypes
        return relative_distance, learned_components

    def get_active_instance_index(self):
        relative_distance_space = self.supervised_metric()[0]
        min_relative_distance = [
            torch.min(distance) for distance in relative_distance_space
        ]
        if self.close_to_boundary:
            return self.get_top_p_close(min_relative_distance)
        return self.get_top_p(min_relative_distance)

    def get_top_p_close(self, x):
        return torch.Tensor(x).argmax() if self.top_n else \
            torch.tensor(x[torch.randperm(len(x))[:self.top_p]]).argmax()

    def get_top_p(self, x):
        return torch.Tensor(x).argmin() if self.top_n else \
            torch.tensor(x[torch.randperm(len(x))[:self.top_p]]).argmin()

    def compute_learned_components_stability(self, learned_components_list):

        change = np.subtract(
            learned_components_list[-2],
            learned_components_list[-1]
        )
        if np.linalg.norm(change, self.ord) <= self.early_stopping:
            return True
        return None

    def select_active_learned_instances(self):
        should_continue = True
        selected_instances, learned_components_list, count = [], [], -1
        while should_continue:
            count += 1
            selected_instances.append(self.get_active_instance_index())
            learned_components_list.append(self.supervised_metric()[1])
            if self.warm_start < count and self.compute_learned_components_stability(
                    learned_components_list=learned_components_list):
                should_continue = False
        return torch.Tensor(selected_instances).detach().cpu().numpy()
