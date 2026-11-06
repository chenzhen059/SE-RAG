import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from sklearn.utils import resample
from distutils.util import strtobool

from utils.metrics import metric

plt.switch_backend('agg')
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
        
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def boot_res(preds,labels ):
    n_iterations = 1000
    n_size = len(preds)
    stats = [] 
    res =  np.mean(np.abs(preds - labels), axis=(1, 2))  
    print(res.shape)
    assert len(res) == n_size
    for _ in range(n_iterations):
        sample = resample(res, n_samples=n_size , replace=True )  
        stats.append(np.mean(sample) )
    return stats
    
def bootstraptest(model, test_loader, args, device ):
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x[:, -args.seq_len:, :], 0)
            
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    
    return np.mean(np.abs(preds - trues), axis=(1, 2))  

def test(model, test_data, test_loader, args, device):
    preds = []
    trues = []
    prevs = []
    model.model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
        
            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()
            
            outputs = model.predict_quantiles(context=torch.tensor(batch_x, dtype=torch.float32),
                prediction_length=args.pred_len,
                quantile_levels=[0.5],
                num_samples=5,
            )[1]
            
            # encoder - decoder
            outputs = outputs[:, -args.pred_len:]
            batch_y = batch_y[:, -args.pred_len:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            prev = batch_x[:, -args.seq_len:].cpu()
            
            preds.append(pred)
            trues.append(true)
            prevs.append(prev)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    prevs = torch.cat(prevs, dim=0).numpy()
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)
    
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def test_longer(model, test_data, test_loader, args, device):
    preds = []
    trues = []
    prevs = []
    if args.model == 'ChronosBolt':
        model.model.eval()
    else:
        model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            
        
            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()

            # rolling retrieve and predict
            predictions = []
            remaining = args.pred_len
            while remaining > 0:
                if args.model == 'ChronosBolt':
                    outputs = model.predict_quantiles(context=torch.tensor(batch_x, dtype=torch.float32),
                        prediction_length=args.pred_len,
                        quantile_levels=[0.5],
                        num_samples=5,
                    )[1]
                elif args.model == 'MOMENT':
                    outputs = model(x_enc=batch_x.unsqueeze(1))
                    outputs = outputs.forecast.squeeze(1)

                else:
                    raise ValueError('model error')

                # update
                if predictions == []:
                    predictions = outputs
                else:
                    predictions = torch.cat([predictions, outputs], dim=1)

                outputs = outputs.to(batch_x)

                batch_x = torch.cat([batch_x, outputs], dim=1)
                batch_x = batch_x[:, -args.seq_len:]
                remaining -= outputs.shape[-1]

                if remaining <=0:
                    predictions = predictions[:, :args.pred_len]
                    break
                
            pred = predictions.detach().cpu()
            true = batch_y.detach().cpu()
            
            preds.append(pred)
            trues.append(true)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    if preds.shape != trues.shape:
        preds = preds[:, :, :trues.shape[-1]]
    print('test shape:', preds.shape, trues.shape)
    
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def oracle_candidate_quality(test_loader, args):
    """
    Oracle 候选质量实验：跳过模型，直接计算检索候选未来片段与真实未来的 MSE。

    对每个测试样本，retrieved_seqs 的后 pred_len 步为候选未来片段：
      - TS-only  : 前 top_k 个候选中取最小 MSE
      - Text-only: 后 top_k_text 个候选中取最小 MSE
      - Combined : 全部 top_k + top_k_text 个候选中取最小 MSE

    最后在整个测试集上取平均并打印。
    top_k_text=0 时跳过 text / combined 输出。
    """
    top_k      = args.top_k
    top_k_text = getattr(args, 'top_k_text', 0)
    quantile_levels = [0.50, 0.75, 0.90, 0.95, 0.99]

    def _print_mse_quantiles(prefix, values):
        values = values.float().flatten()
        q_tensor = torch.tensor(quantile_levels, device=values.device)
        qs = torch.quantile(values, q_tensor)
        parts = [f"p{int(q * 100):02d}={v.item():.6f}" for q, v in zip(quantile_levels, qs)]
        print(f"  {prefix}: " + "  ".join(parts))

    ts_min_mses, text_min_mses, comb_min_mses, comb_min_mses_source = [], [], [], []
    ts_rank_mses   = [[] for _ in range(top_k)]
    text_rank_mses = [[] for _ in range(top_k_text)]

    for batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances, self_error_feat in tqdm(test_loader, desc="oracle"):
        # batch_y      : (B, pred_len, 1) or (B, pred_len)
        # retrieved_seqs: (B, total_k, seq_len + pred_len)
        pred_len = args.pred_len
        batch_y = batch_y.float()                                   # (B, pred_len, 1) or (B, pred_len)
        if batch_y.dim() == 3:
            batch_y = batch_y.squeeze(-1)                           # (B, pred_len)

        # future portion of each candidate: last pred_len steps
        futures = retrieved_seqs.float()[..., -pred_len:]           # (B, total_k, pred_len)
        true    = batch_y.unsqueeze(1)                              # (B, 1, pred_len)

        mse_per_cand = ((futures - true) ** 2).mean(dim=-1)        # (B, total_k)

        actual_k = mse_per_cand.shape[1]
        actual_top_k = min(top_k, actual_k)
        actual_top_k_text = max(0, actual_k - actual_top_k)

        # TS-only
        if actual_top_k > 0:
            ts_mse_per = mse_per_cand[:, :actual_top_k]              # (B, actual_top_k)
            ts_min_mses.append(ts_mse_per.min(dim=-1).values)       # (B,)
            for r in range(actual_top_k):
                ts_rank_mses[r].append(ts_mse_per[:, r])

        # Text-only & combined
        if actual_top_k_text > 0:
            text_mse_per = mse_per_cand[:, actual_top_k:actual_top_k + actual_top_k_text]
            text_min_mses.append(text_mse_per.min(dim=-1).values)
            comb_mse, comb_argmin = mse_per_cand[:, :actual_top_k + actual_top_k_text].min(dim=-1)
            comb_min_mses.append(comb_mse)
            comb_min_mses_source.append(comb_argmin)
            for r in range(actual_top_k_text):
                text_rank_mses[r].append(text_mse_per[:, r])

    print(f"\n[Oracle Candidate Quality]")
    if top_k > 0:
        ts_min_all = torch.cat(ts_min_mses)
        ts_candidate_all = torch.cat([torch.cat(rank_vals) for rank_vals in ts_rank_mses if len(rank_vals) > 0])
        print(f"  TS-only   top{top_k}  min-MSE  (avg over test): {ts_min_all.mean().item():.6f}")
        _print_mse_quantiles("TS min-MSE quantiles", ts_min_all)
        _print_mse_quantiles("TS all-candidate MSE quantiles", ts_candidate_all)
        # per-rank avg MSE for TS
        print(f"  TS per-rank avg-MSE:")
        for r in range(top_k):
            if len(ts_rank_mses[r]) > 0:
                avg = torch.cat(ts_rank_mses[r]).mean().item()
                print(f"    rank-{r+1:2d}: {avg:.6f}")

    if top_k_text > 0 and len(text_min_mses) > 0:
        text_min_all = torch.cat(text_min_mses)
        text_candidate_all = torch.cat([torch.cat(rank_vals) for rank_vals in text_rank_mses if len(rank_vals) > 0])
        comb_min_all = torch.cat(comb_min_mses)
        print(f"  Text-only top{top_k_text}  min-MSE  (avg over test): {text_min_all.mean().item():.6f}")
        _print_mse_quantiles("Text min-MSE quantiles", text_min_all)
        _print_mse_quantiles("Text all-candidate MSE quantiles", text_candidate_all)
        print(f"  Combined  top{top_k + top_k_text}  min-MSE  (avg over test): {comb_min_all.mean().item():.6f}")
        _print_mse_quantiles("Combined min-MSE quantiles", comb_min_all)
        # per-rank avg MSE for Text
        print(f"  Text per-rank avg-MSE:")
        for r in range(top_k_text):
            if len(text_rank_mses[r]) > 0:
                avg = torch.cat(text_rank_mses[r]).mean().item()
                print(f"    rank-{r+1:2d}: {avg:.6f}")

        # 来源统计
        all_argmin = torch.cat(comb_min_mses_source)
        from_ts   = (all_argmin < top_k).sum().item()
        from_text = (all_argmin >= top_k).sum().item()
        total     = len(all_argmin)
        print(f"\n[Oracle Best-Candidate Source]  (combined top{top_k + top_k_text})")
        print(f"  From TS  : {from_ts:6d} / {total}  ({100 * from_ts / total:.1f}%)")
        print(f"  From Text: {from_text:6d} / {total}  ({100 * from_text / total:.1f}%)")


def test_retrieve(model, test_data, test_loader, args, device):
    preds = []
    trues = []
    prevs = []
    model.eval()
    if not os.path.exists(f"./results/retrieve_visulization/{args.model_id.split('_')[0]}"):
        os.makedirs(f"./results/retrieve_visulization/{args.model_id.split('_')[0]}")
    
    if not os.path.exists(f"./results/retrieve_visulization/{args.model_id.split('_')[0]}_vis"):
        os.makedirs(f"./results/retrieve_visulization/{args.model_id.split('_')[0]}_vis")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            # unpack — loaders now return 7-tuples with self_error_feat
            if len(batch) == 7:
                batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances, self_error_feat = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark, retrieved_seqs, distances = batch
                self_error_feat = None

            batch_x = batch_x.float().to(device).squeeze()
            batch_y = batch_y.float().to(device).squeeze()
            retrieved_seqs = retrieved_seqs.float().to(device)
            distances = distances.float().to(device)
            if self_error_feat is not None:
                self_error_feat = self_error_feat.float().to(device)

            if args.model == 'ChronosBoltRetrieve':
                outputs = model(context = batch_x,
                                target = batch_y,
                                retrieved_seq = retrieved_seqs, 
                                distances = distances,
                                top_k_ts = args.top_k,
                                self_error_feat = self_error_feat)        # ChronosBoltOutput
                outputs = outputs.quantile_preds.to(batch_x)
                central_idx = torch.abs(torch.tensor(quantiles) - 0.5).argmin()
                outputs = outputs[:, central_idx]
            elif args.model == 'MOMENTRetrieve':
                if getattr(model, 'use_retrieval', True):
                    outputs = model(
                        x_enc=batch_x.float().unsqueeze(1),
                        retrieved_seq=retrieved_seqs.float(),
                        distances=distances.float(),
                        top_k_ts=args.top_k,
                        self_error_feat=self_error_feat,
                    )
                else:
                    outputs = model(x_enc=batch_x.float().unsqueeze(1))
                outputs = outputs.forecast.squeeze(1)
            else:
                raise ValueError('model error')

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            
            preds.append(pred)
            trues.append(true)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)
    
    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def scaler_inverse_transform(scaler, data, id):
    scale = scaler.scale_[id]
    mean = scaler.mean_[id]
    data = data.cpu().numpy()
    data_inv = data * scale + mean
    return data_inv

def retrieve_given_feat_id(retrievers, raw_datas, batch_x, feat_id, top_k, embedding_model, scaler):
    # devide batch_x based on feat_id
    distances = np.array([])
    retrieved_seqs = np.array([])
    unique_ids = feat_id.unique()
    for unique_id in unique_ids:
        idx = (feat_id == unique_id)
        batch_x_sub = batch_x[idx]
        batch_x_sub = scaler_inverse_transform(scaler, batch_x_sub, unique_id)
        query_sub, _ = embedding_model.embed(torch.from_numpy(batch_x_sub).to(feat_id.device))
        query_sub = query_sub[:,-1,:].squeeze().float().numpy()
        distances_sub, _, timestamp_idx_sub = retrievers[unique_id].search(query_sub, top_k=top_k)
        retrieved_seq_sub = np.array([
            [raw_datas[unique_id][idx: idx + 512 + 64] for idx in row] 
            for row in timestamp_idx_sub
        ])
        if distances.size == 0:
            distances = distances_sub
            retrieved_seqs = retrieved_seq_sub
        else:
            distances = np.concatenate([distances, distances_sub], axis=0)
            retrieved_seqs = np.concatenate([retrieved_seqs, retrieved_seq_sub], axis=0)
    distances = torch.from_numpy(distances)
    retrieved_seqs = torch.from_numpy(retrieved_seqs)
    return retrieved_seqs, distances

def test_real_time_retrieve(model, test_data, test_loader, args, device, retrievers, retriever_rawdata, embedding_model):
    preds = []
    trues = []
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, feat_id) in tqdm(enumerate(test_loader)):

            batch_x = batch_x.float().to(device).squeeze(-1)
            batch_y = batch_y.float().to(device).squeeze(-1)
            
            # rolling retrieve and predict
            predictions = []
            remaining = args.pred_len
            while remaining > 0:
                # retrieve
                batch_x_query = batch_x[:, -args.seq_len:].cpu()
                retrieved_seqs, distances = retrieve_given_feat_id(retrievers, retriever_rawdata, batch_x_query, feat_id, args.top_k, embedding_model, test_data.scaler)

                retrieved_seqs = retrieved_seqs.float().to(device)
                distances = distances.float().to(device)
                outputs = model(context = batch_x,
                                retrieved_seq = retrieved_seqs, 
                                distances = distances)                  # ChronosBoltOutput
                outputs = outputs.quantile_preds.to(batch_x)
            
                central_idx = torch.abs(torch.tensor(quantiles) - 0.5).argmin()
                outputs = outputs[:, central_idx]

                # update
                if predictions == []:
                    predictions = outputs
                else:
                    predictions = torch.cat([predictions, outputs], dim=1)
                batch_x = torch.cat([batch_x, outputs], dim=1)
                remaining -= outputs.shape[-1]
                
                if remaining <=0:
                    predictions = predictions[:, :args.pred_len]
                    break

            pred = predictions.detach().cpu()
            true = batch_y.detach().cpu()
            preds.append(pred)
            trues.append(true)

    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

def multi_test_retrieve(model, test_loaders, args, device, dataset_probabilities):
    test_results = []
    for loader, prob in zip(test_loaders, dataset_probabilities):
        mse, mae = test_retrieve(model, None, loader, args, device, 1)
        test_results.append((mse, mae, prob))
    return test_results

def get_borders(dataset_name, seq_len, total_length=None):
    """
    get the border index according to the dataset name, sequence length and set type.
    """
    if 'ETTh' in dataset_name:
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif 'ETTm' in dataset_name:
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    elif dataset_name in ['electricity', 'exchange_rate', 'weather', 'traffic']:
        if total_length is None:
            raise ValueError("need to provide total_length for {}".format(dataset_name))
        num_train = int(total_length * 0.7)
        num_test  = int(total_length * 0.2)
        num_vali  = total_length - num_train - num_test
        border1s = [0, num_train - seq_len, total_length - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, total_length]
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))
    return border1s, border2s
