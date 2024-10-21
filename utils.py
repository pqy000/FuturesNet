import torch
import numpy as np
import torch.optim as optim
import torch
import os
from torch import nn
from functools import partial
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.preprocessing import minmax_scale

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

# 1*10*15
class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, args, num_classes=5):
        self.device = device
        self.P = args.window
        self.h = args.horizon
        file_name = file_name
        start_time, end_time = self.get_prediction_interval(args.model)
        self.rawdat = np.load(file_name)[start_time:end_time]
        # self.rawdat = self.rawdat
        self.n, self.m = self.rawdat.shape[0], self.rawdat.shape[1]-1

        self.dat = self.rawdat[:,:-1]
        self.label = self.rawdat[:,-1] + 2
        self.num_classes = num_classes
        self.scale = np.ones(self.m)
        self._normalized(args.normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        self.scale = torch.as_tensor(self.scale, device=device, dtype=torch.float)
        self.scale = Variable(self.scale)

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.
        if (normalize == 0):
            self.dat = self.rawdat
        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)
    def get_prediction_interval(self, model_name, length = 1000):
        # 定义每个模型对应的预测区间
        model_intervals = {
            "CNN": (length*2, length*3),
            "GRU": (length*3, length*4),
            "Transformer": (length*2, length*3),
            "LSTNet": (length*4, length*5),
            "FuturesNet": (length * 3, length * 4),
        }
        if model_name in model_intervals:
            start_time, end_time = model_intervals[model_name]
            return start_time, end_time
        else:
            raise ValueError(f"Model '{model_name}' not supported. Choose from {list(model_intervals.keys())}.")

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m), device=self.device)
        Y = torch.zeros((n, 1), device=self.device)

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.as_tensor(self.dat[start:end, :], device=self.device)
            Y[i, :] = torch.as_tensor(self.label[idx_set[i]],device=self.device)
        Y = torch.squeeze(Y, 1).to(device=self.device, dtype=torch.int64)
        return [X, Y]
    # 10 * 15

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length,device=self.device)
        else:
            index = torch.as_tensor(range(length),device=self.device,dtype=torch.long)
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]

            yield Variable(X), Variable(Y)
            start_idx += batch_size

def RescaledSaliency(mask, isTensor=True):
    if(isTensor):
        saliency = np.absolute(mask.data.cpu().numpy())
    else:
        saliency = np.absolute(mask)
    saliency  = saliency.reshape(mask.shape[0], -1)
    rescaledsaliency = minmax_scale(saliency, axis=1)
    rescaledsaliency = rescaledsaliency.reshape(mask.shape)
    return rescaledsaliency

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve
     after a given patience.

    Args:
        patience (int): How long to wait after last time validation loss improved.
            Default to 7
        verbose (bool): If True, prints a message for each validation loss
            improvement. Default to False
        delta (float): Minimum change in the monitored quantity to qualify as an
            improvement. Default to 0.

    """

    def __init__(self, monitor='val_loss', patience=7, delta=0, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor = monitor
        self.delta = delta
        self.value = np.Inf

    def __call__(self, value, model):
        if self.monitor == 'val_loss':
            score = -value
        elif self.monitor == 'val_acc':
            score = value
        else:
            print('Error in initial monitor')

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(value, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(value, model)
            self.counter = 0

    def save_checkpoint(self, value, model):
        """Saves checkpoint when validation loss decrease.

        Args:
            value (float): The value of new validation loss.
            model (model): The current better model.
        """

        if self.verbose:
            print(f'{self.monitor} updated:({self.value:.6f} --> {value:.6f}).')
        self.best_model = model
        self.value = value


class aleatoric_loss(nn.Module):
    """The negative log likelihood (NLL) loss.

    Args:
        gt: the ground truth
        pred_mean: the predictive mean
        logvar: the log variance
    Attributes:
        loss: the nll loss result for the regression.
    """

    def __init__(self):
        super(aleatoric_loss, self).__init__()

    def forward(self, gt, pred_mean, logvar):
        loss = torch.sum(0.5 * (torch.exp((-1) * logvar)) *
                         (gt - pred_mean) ** 2 + 0.5 * logvar)
        return loss


class mmd_loss(nn.Module):
    """The mmd loss.

    Args:
        source_features: the ground truth
        target_features: the prediction value
    Attributes:
        loss_value: the nll loss result for the regression.
    """

    def __init__(self):
        super(mmd_loss, self).__init__()

    def forward(self, source_features, target_features):

        sigmas = [
            1, 4, 8, 16, 24, 32, 64
        ]
        if source_features.is_cuda:
            gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
            )
        else:
            source_features = source_features.cpu()
            target_features = target_features.cpu()
            gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas=Variable(torch.FloatTensor(sigmas))
            )

        loss_value = self.maximum_mean_discrepancy(
            source_features, target_features, kernel=gaussian_kernel)
        loss_value = loss_value

        return loss_value

    def pairwise_distance(self, x, y):

        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[1] != y.shape[1]:
            raise ValueError('The number of features should be the same.')

        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)

        return output

    def gaussian_kernel_matrix(self, x, y, sigmas):

        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_)

        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(self, x, y, kernel=gaussian_kernel_matrix):

        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))

        return cost

def makeOptimizer(params, args):
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError("Invalid optim method: " + args.method)
    return optimizer

def RescaledSaliency(mask, isTensor=True):
    if(isTensor):
        saliency = np.absolute(mask.data.cpu().numpy())
    else:
        saliency = np.absolute(mask)
    saliency  = saliency.reshape(mask.shape[0], -1)
    rescaledsaliency = minmax_scale(saliency, axis=1)
    rescaledsaliency = rescaledsaliency.reshape(mask.shape)
    return rescaledsaliency

def save_model(model, model_path):
    print("saving the model to %s" % model_path)
    try:
        torch.save(model.state_dict(), model_path)
    except:
        torch.save(model, model_path)

def load_model(model, path):
    """Load Pytorch model.

    Args:
        model (pytorch model): The initialized pytorch model.
        model_path (string or path): Path for loading model.

    Returns:
        model: The loaded model.
    """
    print("loading %s" % path)
    with open(path, 'rb') as f:
        pretrained = torch.load(f, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
    return model

