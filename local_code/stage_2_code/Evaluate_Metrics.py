'''
Concrete Evaluate class for multiclass evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Metrics(evaluate):
    data = None
    # multiclass classification, so binary precision/recall/f1 cannot be used
    averages = ['weighted', 'macro', 'micro']

    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        scores = {'accuracy': accuracy_score(true_y, pred_y)}
        for average in self.averages:
            scores['precision_' + average] = precision_score(true_y, pred_y, average=average, zero_division=0)
            scores['recall_' + average] = recall_score(true_y, pred_y, average=average, zero_division=0)
            scores['f1_' + average] = f1_score(true_y, pred_y, average=average, zero_division=0)
        return scores
