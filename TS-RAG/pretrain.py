import os
import time
import wandb
import torch
import random
import argparse
import warnings
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.moment import MOMENTPipelineWithRetrieval
from dataset import CustomPretrainDataset, Retriever_for_pretrain
from models.ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval
    
warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='ChronosBoltRetrieve')

parser.add_argument('--model_id', type=str, default='ChronosBoltRetrieve_Pretrain')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# retrieve
parser.add_argument('--embedding_tuning', type=str, default=None)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--embedding_model_type', type=str, default='chronos')
parser.add_argument('--retrieve_lookback_length', type=int, default=64)
parser.add_argument('--retrieval_database_path', type=str, default='../database/pretrain/retrieval_database_512.parquet')

# text KNN
parser.add_argument('--text_knn_dir', type=str, default=None,
                    help='Embeddings/pretrain_query 目录，存放 text_knn_top{k}_chunk*.npz')
parser.add_argument('--top_k_text', type=int, default=0,
                    help='文本近邻数量，0=不使用')

# self-error features
parser.add_argument('--self_error_feat_dir', type=str, default=None,
                    help='self_error_features/pretrain/ 目录，存放 features_chunk*.npz；None=不使用')

# oracle
parser.add_argument('--oracle', action='store_true',
                    help='Oracle 候选质量实验：跳过训练，仅统计训练集候选未来片段的 min-MSE 与来源比例')

# augment
parser.add_argument('--augment_mode', type=str, default='moe2')

# model
parser.add_argument('--model', type=str, default='ChronosBoltRetrieve')
parser.add_argument('--freeze_chronos_bolt', action='store_true', help="freeze the params of chronos-bolt.")
parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/base/')
parser.add_argument('--context_length', type=int, default=512)
parser.add_argument('--prediction_length', type=int, default=64)
parser.add_argument('--finetune_moment_head', action='store_true',
                    help='For MOMENTRetrieve: train only the forecasting head and save it.')
parser.add_argument('--moment_head_checkpoint', type=str, default='./checkpoints/moment_head/head.pth',
                    help='Path to save/load the MOMENT forecasting head checkpoint.')

# pretrain
parser.add_argument('--data_path', type=str, default='../datasets/pretrain/50m-with-retrieval_512', help='pretrain data path')
parser.add_argument('--max_files', type=int, default=None, help='only use first N parquet files for training (None = all)')
parser.add_argument('--train_steps', type=int, default=200_000)
parser.add_argument('--evaluation_steps', type=int, default=10_000)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--tmax', type=int, default=20)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--shuffle_buffer_length', type=int, default=100_000)
parser.add_argument('--grad_clip_value', type=float, default=1.0)

# gpu
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--gpu_loc', type=int, default=0, help='main gpu location')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)


args = parser.parse_args()

# init wandb project
wandb.init(project=f'{args.model}_Pretrain', name=args.model_id)
wandb.config.update(args)


device = 'cuda:'+str(args.gpu_loc)

time_now = time.time()

## load model, optimizer
config = AutoConfig.from_pretrained(args.pretrained_model_path)
if args.model == 'ChronosBolt':
    model = ChronosBoltModelForForecasting.from_pretrained(args.pretrained_model_path, config=config)
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
elif args.model == 'ChronosBoltRetrieve':
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(args.pretrained_model_path, config=config, augment=args.augment_mode)
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.offset_mlp, model.mha, model.ffn, model.gate_layer, model.encode_mlp_text, model.mha_text, model.ffn_text, model.gate_layer_text])
    if 'gate' in args.augment_mode:
        model.init_extra_weights([model.gate_layer, model.gate_linear1, model.gate_linear2])
elif args.model == 'MOMENTRetrieve':
    MOMENT_MODEL_PATH = "AutonLab/MOMENT-1-large"
    model = MOMENTPipelineWithRetrieval.from_pretrained(MOMENT_MODEL_PATH,
                                           model_kwargs={
                                               'task_name': 'forecasting',
                                               'forecast_horizon': 64,
                                           })
    model.init()
    if args.finetune_moment_head:
        print(f'Finetuning MOMENT head; will save to {args.moment_head_checkpoint}')
    else:
        if os.path.exists(args.moment_head_checkpoint):
            model.head.load_state_dict(torch.load(args.moment_head_checkpoint, map_location='cpu'))
            print(f'Loaded MOMENT head from {args.moment_head_checkpoint}')
        else:
            exit(f'MOMENT head checkpoint not found: {args.moment_head_checkpoint}. Run with --finetune_moment_head first.')
    if 'moe' in args.augment_mode and not args.finetune_moment_head:
        model.init_extra_weights([model.encode_mlp, model.offset_mlp, model.mha, model.ffn, model.gate_layer, model.encode_mlp_text, model.mha_text, model.ffn_text, model.gate_layer_text, model.project_before_fusion, model.project_after_fusion])
    criterion = nn.MSELoss().to(device)
else:
    print('model error')
    exit()
print(f'{args.model} model loaded')

model.to(device)
if args.use_multi_gpu:
    args.devices = [int(i) for i in args.devices.split(',')]
    model = nn.DataParallel(model, device_ids=args.devices)
    
params = model.parameters()

if args.optimizer == 'adam':
    model_optim = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    model_optim = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

# freeze params
if args.model == 'MOMENTRetrieve' and args.finetune_moment_head:
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        param.requires_grad = 'head' in name
elif args.freeze_chronos_bolt:
    layers_to_unfreeze = ['gate_layer', 'encode_mlp', 'offset_mlp', 'mha', 'ffn', 'gate_layer_text', 'encode_mlp_text', 'mha_text', 'ffn_text', 'project_before_fusion', 'project_after_fusion']
    if args.augment_mode == 'moe3':
        if args.model == 'ChronosBoltRetrieve':
            layers_to_unfreeze.append('output_patch_embedding')
        elif args.model == 'MOMENTRetrieve':
            # import pdb; pdb.set_trace()
            layers_to_unfreeze.append('head')
    elif args.augment_mode == 'gate':
        layers_to_unfreeze.append('gate_linear1')
        layers_to_unfreeze.append('gate_linear2')

    for param in model.parameters():
        param.requires_grad = False
    # unfreeze the specified layers
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in layers_to_unfreeze)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

# retrieval already done, do not need to load the embedding model
embedding_model = None

# load retriever
retriever = Retriever_for_pretrain(
    retrieval_database_path=args.retrieval_database_path,
    dimension=768,
    embedding_model=embedding_model,
)
retriever.build_index()

## load data
dataset = CustomPretrainDataset(
    args.data_path, 
    retriever=retriever, 
    mode='training',
    drop_prob=args.drop_prob,
    context_length=args.context_length,
    prediction_length=args.prediction_length,
    retrieve_lookback_length=args.retrieve_lookback_length,
    top_k=args.top_k,
    max_files=args.max_files,
    text_knn_dir=args.text_knn_dir if args.top_k_text > 0 else None,
    top_k_text=args.top_k_text,
    text_db_rawdata=retriever.whole_seq if args.top_k_text > 0 else None,
    self_error_feat_dir=args.self_error_feat_dir,
).shuffle(shuffle_buffer_length=args.shuffle_buffer_length)

train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

# ── Oracle 候选质量实验（--oracle 时跳过训练直接退出）────────────────────────
if args.oracle:
    from tqdm import tqdm as _tqdm
    oracle_steps   = args.train_steps   # 用相同步数扫描，保持可比性
    pred_len       = args.prediction_length
    top_k          = args.top_k
    top_k_text     = args.top_k_text
    quantile_levels = [0.50, 0.75, 0.90, 0.95, 0.99]

    def _print_mse_quantiles(prefix, values):
        arr = values.float().flatten().cpu().numpy()
        qs = np.quantile(arr, quantile_levels)
        parts = [f"p{int(q * 100):02d}={v:.6f}" for q, v in zip(quantile_levels, qs)]
        print(f"  {prefix}: " + "  ".join(parts))

    comb_min_mses  = []
    comb_argmins   = []
    ts_min_mses    = []
    text_min_mses  = []
    ts_rank_mses   = [[] for _ in range(top_k)]
    text_rank_mses = [[] for _ in range(top_k_text)]

    print(f"[oracle-pretrain] scanning {oracle_steps} batches …")
    for _i, batch in enumerate(_tqdm(train_loader, total=oracle_steps, desc="oracle")):
        if _i >= oracle_steps:
            break

        # ── 构建 retrieved_seqs（和训练完全一致）──────────────────────────
        _retrieved = torch.tensor(retriever.whole_seq[batch['indices']])  # (B, top_k, L)
        if top_k_text > 0 and batch['text_indices'].shape[-1] > 0:
            _text = torch.tensor(retriever.whole_seq[batch['text_indices']])  # (B, top_k_text, L)
            _retrieved = torch.cat([_retrieved, _text], dim=1)               # (B, top_k+top_k_text, L)

        # ── 真实未来：batch['y'] shape (B, pred_len) ─────────────────────
        _true = batch['y'].float().unsqueeze(1)                              # (B, 1, pred_len)

        # ── 候选未来：retrieved_seqs 最后 pred_len 步 ─────────────────────
        _futures = _retrieved.float()[..., -pred_len:]                       # (B, total_k, pred_len)

        _mse_per = ((_futures - _true) ** 2).mean(dim=-1)                   # (B, total_k)
        _comb_mse, _comb_argmin = _mse_per.min(dim=-1)                      # (B,)

        actual_k = _mse_per.shape[1]
        actual_top_k = min(top_k, actual_k)
        actual_top_k_text = max(0, actual_k - actual_top_k)

        if actual_top_k > 0:
            _ts_mse_per = _mse_per[:, :actual_top_k]
            ts_min_mses.append(_ts_mse_per.min(dim=-1).values)
            for r in range(actual_top_k):
                ts_rank_mses[r].append(_ts_mse_per[:, r])
        if actual_top_k_text > 0:
            _text_mse_per = _mse_per[:, actual_top_k:actual_top_k + actual_top_k_text]
            text_min_mses.append(_text_mse_per.min(dim=-1).values)
            for r in range(actual_top_k_text):
                text_rank_mses[r].append(_text_mse_per[:, r])
        comb_min_mses.append(_comb_mse)
        comb_argmins.append(_comb_argmin)

    comb_min_mses = torch.cat(comb_min_mses)
    comb_argmins  = torch.cat(comb_argmins)
    total_k       = top_k + top_k_text

    from_ts   = (comb_argmins < top_k).sum().item()
    from_text = (comb_argmins >= top_k).sum().item() if top_k_text > 0 and len(text_min_mses) > 0 else 0
    total     = len(comb_argmins)

    print(f"\n[Oracle Candidate Quality — pretrain train split]")
    if top_k > 0:
        ts_min_all = torch.cat(ts_min_mses)
        ts_candidate_all = torch.cat([torch.cat(rank_vals) for rank_vals in ts_rank_mses if len(rank_vals) > 0])
        print(f"  TS-only   top{top_k}  min-MSE (avg): {ts_min_all.mean().item():.6f}")
        _print_mse_quantiles("TS min-MSE quantiles", ts_min_all)
        _print_mse_quantiles("TS all-candidate MSE quantiles", ts_candidate_all)
        print(f"  TS per-rank avg-MSE:")
        for r in range(top_k):
            if len(ts_rank_mses[r]) > 0:
                print(f"    rank-{r+1:2d}: {torch.cat(ts_rank_mses[r]).mean().item():.6f}")
    if top_k_text > 0 and len(text_min_mses) > 0:
        text_min_all = torch.cat(text_min_mses)
        text_candidate_all = torch.cat([torch.cat(rank_vals) for rank_vals in text_rank_mses if len(rank_vals) > 0])
        print(f"  Text-only top{top_k_text}  min-MSE (avg): {text_min_all.mean().item():.6f}")
        _print_mse_quantiles("Text min-MSE quantiles", text_min_all)
        _print_mse_quantiles("Text all-candidate MSE quantiles", text_candidate_all)
        print(f"  Text per-rank avg-MSE:")
        for r in range(top_k_text):
            if len(text_rank_mses[r]) > 0:
                print(f"    rank-{r+1:2d}: {torch.cat(text_rank_mses[r]).mean().item():.6f}")
    print(f"  Combined  top{total_k}  min-MSE (avg): {comb_min_mses.mean().item():.6f}")
    _print_mse_quantiles("Combined min-MSE quantiles", comb_min_mses)
    print(f"\n[Oracle Best-Candidate Source]  (combined top{total_k})")
    print(f"  From TS  : {from_ts:6d} / {total}  ({100 * from_ts / total:.1f}%)")
    print(f"  From Text: {from_text:6d} / {total}  ({100 * from_text / total:.1f}%)")
    exit(0)


## train
is_first = True 
iter_count = 0
train_loss = []
pbar = tqdm(total=args.train_steps, desc='Training', dynamic_ncols=True)
for i, batch in enumerate(train_loader):
    if i >= args.train_steps:
        print('training finished')
        pbar.close()
        break
    if is_first : 
        print(args.data_path, batch['x'].shape, batch['y'].shape, batch['distances'].shape, batch['indices'].shape)
        is_first=False
        
    # ── Batch-level sample filtering ──────────────────────────────────────
    # 过滤三类问题样本（在CPU上操作，代价极低）
    x_f = batch['x'].float()                       # (B, ctx_len)
    y_f = batch['y'].float()                       # (B, pred_len)
    x_std = x_f.std(dim=-1)                        # (B,)
    y_std = y_f.std(dim=-1)                        # (B,)
    ratio = y_std / (x_std + 1e-8)                 # (B,)

    keep = (
        (x_std >= 1e-3)   &   # 剔除近常数 context
        (x_std <= 8.0)    &   # 剔除极端高方差 context
        (ratio <= 10.0)       # 剔除 target 相对 context 剧变
    )                                              # (B,) bool

    if keep.sum() == 0:
        # 整个 batch 都是坏样本，直接跳过
        continue

    # 只保留合格样本
    if keep.sum() < keep.shape[0]:
        keep_idx = keep.nonzero(as_tuple=True)[0]
        for k in batch:
            batch[k] = batch[k][keep_idx]
    # ── end filtering ──────────────────────────────────────────────────────

    iter_count += 1
    model_optim.zero_grad()
    # ── 时序近邻序列 (B, top_k, seq_len) ──────────────────────────────────
    retrieved_seqs = torch.tensor(retriever.whole_seq[batch['indices']])   # (B, top_k, L)

    # ── 文本近邻序列：拼接到时序近邻之后 ────────────────────────────────────
    if args.top_k_text > 0 and batch['text_indices'].shape[-1] > 0:
        text_retrieved = torch.tensor(
            retriever.whole_seq[batch['text_indices']]                     # (B, top_k_text, L)
        )
        retrieved_seqs = torch.cat([retrieved_seqs, text_retrieved], dim=1) # (B, top_k+top_k_text, L)
        distances_all  = torch.cat(
            [batch['distances'], batch['text_distances']], dim=1            # (B, top_k+top_k_text)
        )
    else:
        distances_all = batch['distances']

    if not args.use_multi_gpu:
        batch['x'] = batch['x'].float().to(device)
        batch['y'] = batch['y'].float().to(device)
        distances_all  = distances_all.float().to(device)
        retrieved_seqs = retrieved_seqs.float().to(device)
    if args.model == 'ChronosBoltRetrieve':
        outputs = model(context = batch['x'].float(), 
                        target = batch['y'].float(),
                        retrieved_seq = retrieved_seqs.float(), 
                        distances = distances_all.float(),
                        top_k_ts = args.top_k,
                        self_error_feat = batch['self_error_feat'].float().to(device))  # (B, K, 5)
    elif args.model == 'MOMENTRetrieve':
        if args.finetune_moment_head:
            outputs = model(x_enc=batch['x'].float().unsqueeze(1))
        else:
            outputs = model(x_enc=batch['x'].float().unsqueeze(1),
                            retrieved_seq=retrieved_seqs.float(),
                            distances=distances_all.float(),
                            top_k_ts=args.top_k,
                            self_error_feat=batch['self_error_feat'].float().to(device))
        outputs = outputs.forecast.squeeze(1)                                                     
        loss = criterion(outputs, batch['y'].float())
    else:
        print('model error')
    if args.model == 'MOMENTRetrieve':
        pass
    else:
        loss = outputs.loss
    loss = loss.mean()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb.log({
            'loss': loss.item(),
            'lr': model_optim.param_groups[0]['lr']
            })

    train_loss.append(loss.item())
    pbar.update(1)
    pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{model_optim.param_groups[0]["lr"]:.2e}'})

    if (i + 1) % args.evaluation_steps == 0:
        print("\titers: {0} | loss: {1:.7f}".format(i + 1, sum(train_loss) / len(train_loss)))
        train_loss = []
        speed = (time.time() - time_now) / iter_count
        print('\tspeed: {:.4f}s/iter'.format(speed))
        iter_count = 0
        time_now = time.time()
        # save model and optimizer
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            save_path = os.path.join(args.checkpoints, args.model_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path,f'model_steps{i}.pth'))
            torch.save(model_optim.state_dict(), os.path.join(save_path, f'optim_steps{i}.pth'))
            if args.model == 'MOMENTRetrieve' and args.finetune_moment_head:
                head_save_dir = os.path.dirname(args.moment_head_checkpoint)
                if head_save_dir:
                    os.makedirs(head_save_dir, exist_ok=True)
                raw_model = model.module if hasattr(model, 'module') else model
                torch.save(raw_model.head.state_dict(), args.moment_head_checkpoint)
                head_step_path = os.path.join(head_save_dir, f'head_steps{i + 1}.pth')
                torch.save(raw_model.head.state_dict(), head_step_path)
                print(f'Saved MOMENT head to {args.moment_head_checkpoint} and {head_step_path}')
            # ── 保存代码快照（仅第一次或每次均覆盖）─────────────────────
            import shutil, glob as _glob
            _src_root = os.path.dirname(os.path.abspath(__file__))  # TS-RAG/TS-RAG/
            _code_dst = os.path.join(save_path, 'code_snapshot')
            os.makedirs(_code_dst, exist_ok=True)
            for _fname in ['pretrain.py', 'dataset.py', 'zeroshot.py']:
                _fpath = os.path.join(_src_root, _fname)
                if os.path.exists(_fpath):
                    shutil.copy2(_fpath, _code_dst)
            for _fname in _glob.glob(os.path.join(_src_root, 'script', 'pretrain*.sh')):
                shutil.copy2(_fname, _code_dst)
            for _fname in _glob.glob(os.path.join(_src_root, 'models', 'ChronosBolt.py')):
                shutil.copy2(_fname, _code_dst)
            for _fname in _glob.glob(os.path.join(_src_root, 'models', 'moment.py')):
                shutil.copy2(_fname, _code_dst)
            # ── end code snapshot ──────────────────────────────────────

        # adjust learning rate
        scheduler.step()
        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

    loss.backward()
    clip_grad_norm_(model.parameters(), args.grad_clip_value)
    model_optim.step()
                
