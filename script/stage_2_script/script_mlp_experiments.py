from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plot_Curves import plot_history, plot_comparison
import numpy as np
import torch
import json
import time

#---- MLP experiments: architecture, activation, loss, optimizer ----
result_folder = '../../result/stage_2_result/'

# every experiment starts from these settings and overrides some of them
default_config = {'hidden_sizes': (512, 256, 128), 'activation': 'relu', 'dropout': 0.0, 'batch_norm': False,
                  'optimizer_name': 'adam', 'learning_rate': 1e-3, 'loss_name': 'cross_entropy',
                  'weight_decay': 0.0, 'lr_scheduler_name': 'none', 'max_epoch': 20, 'batch_size': 128}

experiment_groups = {
    'depth': {
        '1 hidden (128)': {'hidden_sizes': (128,)},
        '2 hidden (256-128)': {'hidden_sizes': (256, 128)},
        '3 hidden (512-256-128)': {'hidden_sizes': (512, 256, 128)},
        '4 hidden (1024-512-256-128)': {'hidden_sizes': (1024, 512, 256, 128)},
    },
    'activation': {
        'ReLU': {'activation': 'relu'},
        'Sigmoid': {'activation': 'sigmoid'},
        'Tanh': {'activation': 'tanh'},
        'GELU': {'activation': 'gelu'},
    },
    'loss': {
        'Cross entropy': {'loss_name': 'cross_entropy'},
        'CE + label smoothing 0.1': {'loss_name': 'label_smoothing'},
        'MSE (softmax vs one-hot)': {'loss_name': 'mse'},
    },
    'optimizer': {
        'SGD (lr 0.01)': {'optimizer_name': 'sgd', 'learning_rate': 1e-2},
        'SGD + momentum (lr 0.01)': {'optimizer_name': 'sgd_momentum', 'learning_rate': 1e-2},
        'RMSprop (lr 1e-3)': {'optimizer_name': 'rmsprop', 'learning_rate': 1e-3},
        'Adam (lr 1e-3)': {'optimizer_name': 'adam', 'learning_rate': 1e-3},
        'AdamW (lr 1e-3, wd 1e-4)': {'optimizer_name': 'adamw', 'learning_rate': 1e-3, 'weight_decay': 1e-4},
    },
    'regularization': {
        'No regularization': {},
        'Dropout 0.2': {'dropout': 0.2},
        'Dropout 0.2 + BatchNorm': {'dropout': 0.2, 'batch_norm': True},
        'Best: Dropout+BN+AdamW+cosine+LS, 40 ep': {'hidden_sizes': (1024, 512, 256), 'dropout': 0.3, 'batch_norm': True,
                                                    'optimizer_name': 'adamw', 'weight_decay': 1e-4,
                                                    'lr_scheduler_name': 'cosine', 'loss_name': 'label_smoothing',
                                                    'max_epoch': 40},
    },
}


def run_experiment(data_obj, name, config):
    np.random.seed(2)
    torch.manual_seed(2)
    method_obj = Method_MLP(name, '', hidden_sizes=config['hidden_sizes'], activation=config['activation'],
                            dropout=config['dropout'], batch_norm=config['batch_norm'])
    for key in ['optimizer_name', 'learning_rate', 'loss_name', 'weight_decay', 'lr_scheduler_name', 'max_epoch', 'batch_size']:
        setattr(method_obj, key, config[key])

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = result_folder + 'MLP_'
    result_obj.result_destination_file_name = 'prediction_result_' + ''.join(c if c.isalnum() else '_' for c in name)

    setting_obj = Setting_Train_Test('pre-partitioned train test', '')
    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    start = time.time()
    scores = setting_obj.load_run_save_evaluate()
    scores['train_time_s'] = time.time() - start
    return scores, method_obj.history


if 1:
    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'

    summary = []
    for group, experiments in experiment_groups.items():
        histories = {}
        for name, overrides in experiments.items():
            print('************ %s: %s ************' % (group, name))
            config = dict(default_config, **overrides)
            scores, history = run_experiment(data_obj, name, config)
            histories[name] = history
            summary.append({'group': group, 'experiment': name, 'config': config, 'scores': scores, 'history': history})
            print({k: round(v, 4) for k, v in scores.items()})
        plot_comparison(histories, 'train_loss', 'Training loss', 'Training loss: ' + group, result_folder + 'compare_%s_loss.png' % group)
        plot_comparison(histories, 'test_acc', 'Test accuracy', 'Test accuracy: ' + group, result_folder + 'compare_%s_test_acc.png' % group)

    best = max(summary, key=lambda s: s['scores']['accuracy'])
    plot_history(best['history'], 'Best MLP: ' + best['experiment'], result_folder + 'MLP_best_curve.png')

    with open(result_folder + 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=1)

    print('************ Summary ************')
    print('| Group | Experiment | Accuracy | F1 (weighted) | F1 (macro) | Precision (macro) | Recall (macro) | Time (s) |')
    print('|---|---|---|---|---|---|---|---|')
    for s in summary:
        sc = s['scores']
        print('| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.0f |' % (s['group'], s['experiment'], sc['accuracy'], sc['f1_weighted'],
                                                                     sc['f1_macro'], sc['precision_macro'], sc['recall_macro'], sc['train_time_s']))
