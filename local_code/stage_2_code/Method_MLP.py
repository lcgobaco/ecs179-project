'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


def get_device():
    # use the Nvidia GPU (CUDA) or the Apple GPU (MPS) when available
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # mini-batch size for each gradient update
    batch_size = 128
    # 'adam', 'adamw', 'sgd', 'sgd_momentum', 'rmsprop'
    optimizer_name = 'adam'
    # 'cross_entropy', 'label_smoothing', 'mse'
    loss_name = 'cross_entropy'
    weight_decay = 0.0
    # 'none' or 'cosine' learning rate schedule
    lr_scheduler_name = 'none'

    # the input portal has 784 features (28x28 pixels) and the output portal has 10 classes (digits 0-9)
    # hidden_sizes defines how many hidden layers there are and the size of each one
    def __init__(self, mName, mDescription, input_size=784, hidden_sizes=(128,), output_size=10,
                 activation='relu', dropout=0.0, batch_norm=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        activation_funcs = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_funcs[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = hidden_size
        # the output layer returns raw scores (logits); softmax is applied inside the loss function
        layers.append(nn.Linear(in_size, output_size))
        self.layers = nn.Sequential(*layers)
        self.output_size = output_size
        self.device = get_device()
        self.history = None

    def forward(self, x):
        '''Forward propagation'''
        # the output is a n x 10 tensor of class scores
        return self.layers(x)

    def get_optimizer(self):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == 'sgd_momentum':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        raise ValueError('unknown optimizer: ' + self.optimizer_name)

    def get_loss_function(self):
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        if self.loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        if self.loss_name == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        if self.loss_name == 'mse':
            # mean squared error between the softmax probabilities and the one-hot labels
            mse = nn.MSELoss()
            return lambda logits, y: mse(torch.softmax(logits, dim=1), nn.functional.one_hot(y, self.output_size).float())
        raise ValueError('unknown loss: ' + self.loss_name)

    # nn.Module.train(mode) is shadowed by the template's train(X, y), so switch
    # between training/evaluation mode (matters for dropout and batch norm) explicitly
    def set_training_mode(self, mode):
        nn.Module.train(self, mode)

    def predict(self, X):
        self.set_training_mode(False)
        with torch.no_grad():
            pred = torch.cat([self.forward(X[i:i + 1024]).argmax(1) for i in range(0, len(X), 1024)])
        return pred.cpu().numpy()

    def train(self, X, y, X_test=None, y_test=None):
        self.to(self.device)
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        scheduler = None
        if self.lr_scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch)
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # move the whole dataset onto the device once, then slice mini-batches from it
        X = torch.as_tensor(np.array(X), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.array(y), dtype=torch.long, device=self.device)
        if X_test is not None:
            X_test = torch.as_tensor(np.array(X_test), dtype=torch.float32, device=self.device)

        self.history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}
        for epoch in range(self.max_epoch):
            self.set_training_mode(True)
            # shuffle and split the training set into mini-batches
            permutation = torch.randperm(len(X), device=self.device)
            epoch_loss = 0.0
            for i in range(0, len(X), self.batch_size):
                batch_index = permutation[i:i + self.batch_size]
                y_pred = self.forward(X[batch_index])
                train_loss = loss_function(y_pred, y[batch_index])

                optimizer.zero_grad()
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # update the variables according to the optimizer and the gradients
                optimizer.step()
                epoch_loss += train_loss.item() * len(batch_index)
            if scheduler is not None:
                scheduler.step()

            accuracy_evaluator.data = {'true_y': y.cpu().numpy(), 'pred_y': self.predict(X)}
            train_acc = accuracy_evaluator.evaluate()
            test_acc = None
            if X_test is not None:
                accuracy_evaluator.data = {'true_y': y_test, 'pred_y': self.predict(X_test)}
                test_acc = accuracy_evaluator.evaluate()
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_loss / len(X))
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            print('Epoch:', epoch + 1, 'Loss: %.4f' % (epoch_loss / len(X)), 'Train Acc: %.4f' % train_acc,
                  '' if test_acc is None else 'Test Acc: %.4f' % test_acc)

    def test(self, X):
        # instances will get the labels corresponding to the largest class score
        X = torch.as_tensor(np.array(X), dtype=torch.float32, device=self.device)
        return self.predict(X)

    def run(self):
        print('method running...')
        print('--start training...')
        # the test set is only used to record the test accuracy curve, never for the gradient updates
        self.train(self.data['train']['X'], self.data['train']['y'], self.data['test']['X'], self.data['test']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}
