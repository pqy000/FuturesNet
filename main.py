import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as opt
import os
import warnings
warnings.filterwarnings("ignore")
from utils import Data_utility, makeOptimizer
from tqdm import tqdm
from models import lob, LSTNet, FuturesNet, CNN, GRU, Transformer
from trainer import origin_train, origin_eval, evaluation, calibration_train, calibration_val
from utils import EarlyStopping, save_model, load_model, RescaledSaliency
from pathlib import Path
import wandb
import json


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--id', type=int, default=300, help='location of the data file')
    parser.add_argument('--year', type=int, default=2020, help='location of the data file')
    parser.add_argument('--model', type=str, default='FuturesNet', help='')
    parser.add_argument('--window', type=int, default=16, help='window size')
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--highway_window', type=int, default=3, help='The window size')
    parser.add_argument('--n_rnns', type=int, default=1, help='depth of the RNN model')
    parser.add_argument('--rnn_hid_size', type=int, default=50)
    parser.add_argument('--rnn_layers', type=int, default=1, help='number of RNN hidden layers')
    parser.add_argument('--fc_hid_size', type=int, default=24)
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--output_fun', type=str, default='tanh')
    parser.add_argument('--hidRNN', type=int, default=64, help='number of RNN hidden units each layer')
    parser.add_argument('--hidCNN', type=int, default=64, help='number of CNN hidden units (channels)')
    parser.add_argument('--CNN_kernel', type=int, default=3, help='the kernel size of the CNN layers')
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', type=str, default="1", help='GPU device id to use')
    parser.add_argument('--model_path', type=str, default='./save', help='path to save checkpoints (default: None)')
    parser.add_argument('--patience', type=int, default=50, help='model path')
    parser.add_argument('--delta', type=int, default=0, help='patience')
    parser.add_argument('--verbose', type=bool, default=False, help='print patience information')
    parser.add_argument('--grad_clip', type=int, default=10, help='clip gradient')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='clip gradient')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--label_smoothing', type=bool, default=False, help='smoothing')
    parser.add_argument('--smoothing', type=float, default=0.08, help='smoothing')
    # parser.add_argument('--length', type=int, default=5000, help='smoothing')
    return parser.parse_args()

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"GPU device ID: {torch.cuda.current_device()}")
    if args.seed:
        torch.cuda.manual_seed(args.seed)

    file_name = "newdata/" + str(args.id) + "_" + str(args.year) + ".npy"
    print(file_name)
    print(str(args.seed))
    Data = Data_utility(file_name, 0.6, 0.2, device, args)
    wandb.init(project="futures"+str(args.id), mode="disabled", config=args, group=str(args.year), name=args.model)
    args.wb = wandb

    model = eval(args.model)
    model = model.Model(args, Data).to(device)
    model.eval()
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)

    optim = makeOptimizer(model.parameters(), args)
    early_stopping_hnn = EarlyStopping(patience=args.patience,  delta=args.delta, verbose=args.verbose)
    weight = [0.3, 0.1, 0.1, 0.2, 0.3]
    class_weights = torch.FloatTensor(weight).cuda()
    criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)

    scheduler = opt.lr_scheduler.ExponentialLR(optim, gamma=0.99)
    print('Training start')
    best_acc = 0
    best_results = {
        'mae': None,
        'mse': None,
        'rmse': None,
        'r2': None,
        'val_loss': None,
        'epoch': None
    }
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # scheduler.step()
        train_loss = origin_train(Data, Data.train[0], Data.train[1], model, optim, criterion, args)
        val_loss, val_acc, result = origin_eval(Data, Data.test[0], Data.test[1], model, criterion, args)
        print('epoch {:2d} | time used: {:5.2f}s | train_loss {:4.3f} | valid loss {:4.3f} | valid acc {:4.3f} '
              '| precision {:4.3f} | recall {:4.3f} | f1 {:4.2f} | sharp {:4.2f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_acc, result['precision'], result['recall'], result['f1_score'], result['sharp_value']) )
        args.wb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1_score'],
            'sharp_value': result['sharp_value'],
        })

        # Save best results
        if val_acc > best_acc:
            best_acc = val_acc
            best_results['best_acc'] = val_acc
            best_results['precision'] = result['precision']
            best_results['recall'] = result['recall']
            best_results['f1_score'] = result['f1_score']
            best_results['val_loss'] = val_loss
            best_results['epoch'] = epoch
            best_results['sharp_value'] = result['sharp_value']

        early_stopping_hnn(val_loss, model)
        if early_stopping_hnn.early_stop:
            print("Early stopping for the phase")
            break
        if epoch % 10 == 0:
            save_path = os.path.join(args.model_path, str(args.year) + "_" + str(args.id), args.model)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path,  f'checkpoint{epoch}.pt')
            save_model(model, save_file)

    results = {
        'best_acc': best_acc,
        'mae': best_results['mae'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1_score': result['f1_score'],
        'val_loss': best_results['val_loss'],
        'epoch': best_results['epoch'],
        'sharp_value': best_results['sharp_value']
    }
    args.wb.log(results)
    results_file = os.path.join(save_path, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
