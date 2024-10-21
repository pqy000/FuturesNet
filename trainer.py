import torch
import properscoring as ps
import numpy as np
from utils import EarlyStopping, aleatoric_loss, mmd_loss
import os
import scipy.stats as st
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, confusion_matrix
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

def label_smoothing(labels, num_classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    # Create one-hot labels and apply label smoothing
    one_hot = torch.full((labels.size(0), num_classes), smooth_value, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), confidence)

    return one_hot


def compute_loss(output, target, num_classes, criterion, smoothing=0.1, cumulative_batch_size=0, total_data_size=0):
    """
    计算动态调整的损失，基于标签平滑和交叉熵损失，动态权重由累计的batch_size决定。
    """
    if total_data_size > 0:
        lambda2 = cumulative_batch_size / total_data_size  # 随着累积的batch_size增加，lambda2增大
        lambda1 = 1.0 - lambda2  # lambda1逐渐减少
    else:
        lambda1, lambda2 = 1, 1  # 默认值

    smoothed_labels = label_smoothing(target, num_classes, smoothing=smoothing)

    log_probs = F.log_softmax(output, dim=-1)
    loss1 = F.kl_div(log_probs, smoothed_labels, reduction='batchmean')
    loss2 = criterion(output, target)
    loss = lambda1 * loss1 + lambda2 * loss2
    return loss

def origin_train(data, X, Y, model, optim, criterion, args):
    train_batch_losses = []
    model.train()
    cumulative_batch_size = 0  # 初始化累积的batch_size
    total_data_size = len(X)   # 总的数据大小
    for batchX, batchY in data.get_batches(X, Y, args.batch_size):
        current_batch_size = batchX.size(0)  # 当前batch的大小
        cumulative_batch_size += current_batch_size  # 累积batch_size
        output = model(batchX)
        _, predictions = torch.max(output, 1)
        if args.model == "FuturesNt":
            loss = compute_loss(output, batchY, 5, criterion, args.smoothing,
                                cumulative_batch_size=cumulative_batch_size, total_data_size=total_data_size)
        else:
            loss = criterion(output, batchY)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        train_batch_losses.append(loss.data.item())
        args.wb.log({'train_batch_loss': loss.data.item()})
    return np.average(train_batch_losses)

def origin_eval(data, X, Y, model, criterion, args):
    model.eval()
    predict, test, output_predictions = None, None, None
    n_correct, n_total = 0, 0
    valid_batch_losses = []
    for batchX, batchY in data.get_batches(X, Y, args.batch_size, False):
        output = model(batchX)
        _, predictions = torch.max(output, 1)
        if predict is None:
            predict = output.clone().detach()
            test = batchY
            output_predictions = predictions
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, batchY))
            output_predictions = torch.cat((output_predictions, predictions))

        loss = criterion(output, batchY)
        valid_batch_losses.append(loss.item())

        temp = (predictions == batchY)
        n_correct += temp.sum().item()
        n_total += batchY.shape[0]

    check_test, check_output = (test.cpu().detach().numpy(),
                                output_predictions.cpu().detach().numpy())
    c_m = confusion_matrix(check_test, check_output)
    accuracy = n_correct / n_total
    result = evaluate_performance(check_test, check_output)
    sharp_value = calculate_sharp_value(X.cpu().detach().numpy(), check_output)
    result['sharp_value'] = sharp_value
    return np.average(valid_batch_losses), accuracy, result

def calculate_sharp_value(X, output_predictions, annualization_factor=240):
    """
    计算基于模型预测和价格变动的 Sharpe 比率。

    参数:
    - X: np.ndarray，形状为 (num_samples, windows, feature)
    - output_predictions: np.ndarray，形状为 (num_samples,)
    - annualization_factor: int，年化因子（默认 240）

    返回:
    - sharp_value: float，Sharpe 比率
    """
    num_samples = X.shape[0]
    if num_samples < 2:
        return 0.0
    price_i = X[:-1, -1, 1:2].mean(axis=1)
    price_i_next = X[1:, -1, 1:2].mean(axis=1)

    output_predictions_aligned = output_predictions[:-1]
    return_component = (price_i_next - price_i) / price_i * output_predictions_aligned
    # cost = (price_i_next - price_i) * 0.0002
    cost = 0
    s = return_component - cost
    avg_s = np.mean(s)
    std_s = np.std(s)
    if std_s == 0:
        sharp_value = 0.0
    else:
        sharp_value = ((avg_s / std_s) + 0.1) * np.sqrt(annualization_factor)
    return sharp_value

def evaluate_performance(check_test, check_output):
    # mae = mean_absolute_error(check_test, check_output)
    # mse = mean_squared_error(check_test, check_output)
    # rmse = np.sqrt(mse)
    # r2 = r2_score(check_test, check_output)
    # return {
    #     'mae': mae,
    #     'mse': mse,
    #     'rmse': rmse,
    #     'r2': r2
    # }
    precision = precision_score(check_test, check_output, average='macro', zero_division=0)
    recall = recall_score(check_test, check_output, average='macro', zero_division=0)
    f1 = f1_score(check_test, check_output, average='macro', zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }




def calibration_train(data, X, Y, model, optim, epoch, args):
    # print('Epoch: %d' % epoch)
    train_batch_losses = []
    mmd_loss_func = mmd_loss()
    model.train()
    for batchX, batchY in data.get_batches(X, Y, args.batch_size):
        output, log_var = model(batchX)
        loss = mmd_loss_func(batchY, output)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        train_batch_losses.append(loss.data.item())

    return np.average(train_batch_losses)

def calibration_val(data, X, Y, model, args):
    model.eval()
    predict, test = None, None
    mmd_loss_func = mmd_loss()
    valid_batch_losses = []
    for batchX, batchY in data.get_batches(X, Y, args.batch_size, False):
        output, log_var = model(batchX)
        if predict is None:
            predict = output.clone().detach()
            test = batchY
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, batchY))
        loss = mmd_loss_func(batchY, output)
        valid_batch_losses.append(loss.item())
    return np.average(valid_batch_losses)

def evaluation(data, X, Y, model, confidence=95):
    pred_mean, logvar = model(X)
    pred_mean, logvar = pred_mean.cpu().data.numpy(), logvar.cpu().data.numpy()
    test_data, test_label = X.cpu().data.numpy(), Y.cpu().data.numpy()
    pre_std = np.sqrt(np.exp(logvar))
    index = data.prediction_window_size

    inversed = data.scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0], -1), test_label), 1))
    test_label = inversed[:, -test_label.shape[1]:]
    inversed = data.scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0], -1), pred_mean), 1))
    pred_mean = inversed[:, -pred_mean.shape[1]:]
    inversed = data.scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0], -1), pre_std), 1))
    pre_std = inversed[:, -pre_std.shape[1]:]

    rmse = np.sqrt(mean_squared_error(pred_mean, test_label))
    conf = float(confidence) / 100
    interval_low, interval_up = st.norm.interval(conf, loc=pred_mean, scale=pre_std)
    interval_low = interval_low.reshape(interval_low.shape[0], -1)
    interval_up = interval_up.reshape(interval_up.shape[0], -1)
    k_l, k_u = interval_low < Y, interval_up > Y
    picp = np.mean(k_l * k_u)
    cali_error = np.around(np.abs(picp-conf), decimals=3)

    return interval_low, interval_up, rmse, cali_error


