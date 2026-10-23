from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_2_code.Plot_Curves import plot_history
import numpy as np
import torch

#---- Multi-Layer Perceptron script (baseline) ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'

    # one hidden layer of 128 units, ReLU, Adam, cross entropy
    method_obj = Method_MLP('multi-layer perceptron', '', hidden_sizes=(128,))
    method_obj.max_epoch = 20
    method_obj.learning_rate = 1e-3
    method_obj.batch_size = 128

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('pre-partitioned train test', '')

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print('device:', method_obj.device)
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    scores = setting_obj.load_run_save_evaluate()
    plot_history(method_obj.history, 'MLP baseline (784-128-10)', '../../result/stage_2_result/MLP_baseline_curve.png')
    print('************ Overall Performance ************')
    for metric, score in scores.items():
        print('MLP %s: %.4f' % (metric, score))
    print('************ Finish ************')
    # ------------------------------------------------------
