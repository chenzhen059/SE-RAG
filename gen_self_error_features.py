

import os
import sys
import argparse
import warnings

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, './TS-RAG')

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_NAMES = ['self_mse', 'self_mae', 'residual_mean', 'residual_std', 'residual_max_abs']
N_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_bolt(pretrained_model_path: str, device: str):
    from transformers import AutoConfig
    from models.ChronosBolt import ChronosBoltModelForForecasting

    config = AutoConfig.from_pretrained(pretrained_model_path)
    bolt   = ChronosBoltModelForForecasting.from_pretrained(
        pretrained_model_path, config=config)
    bolt.load_state_dict(
        torch.load(f'{pretrained_model_path}/autogluon_model.pth',
                   map_location='cpu'), strict=False)
    bolt.to(device).eval()
    for p in bolt.parameters():
        p.requires_grad_(False)
    return bolt


def bolt_predict_array(bolt, ctx: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """
    ctx : (N, ctx_len) float32 numpy
    Returns pred : (N, pred_len) float32 numpy  — median quantile
    """
    results = []
    for start in tqdm(range(0, len(ctx), batch_size), desc='bolt_predict', leave=False):
        chunk = torch.tensor(ctx[start:start + batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = bolt(context=chunk)
        if hasattr(out, 'quantile_preds'):
            mid  = out.quantile_preds.shape[1] // 2
            pred = out.quantile_preds[:, mid, :]
        elif hasattr(out, 'samples'):
            pred = out.samples.median(dim=1).values
        else:
            raise ValueError('Unknown ChronosBoltOutput format')
        results.append(pred.cpu().numpy().astype(np.float32))
    return np.concatenate(results, axis=0)


def compute_features_from_lookup(
    pred_batch:   np.ndarray,    # (K, pred_len)
    future_batch: np.ndarray,    # (K, pred_len)
) -> tuple:
    """Returns (feat_raw, feat_local_norm), both (K, 5) float32."""
    resid    = pred_batch - future_batch                  # (K, pred_len)
    self_mse = (resid ** 2).mean(axis=-1)
    self_mae = np.abs(resid).mean(axis=-1)
    res_mean = resid.mean(axis=-1)
    res_std  = resid.std(axis=-1)
    res_max  = np.abs(resid).max(axis=-1)

    feat_raw = np.stack([self_mse, self_mae, res_mean, res_std, res_max],
                        axis=-1).astype(np.float32)       # (K, 5)

    EPS     = 1e-6
    fut_std = future_batch.std(axis=-1)
    fut_var = fut_std ** 2
    scale   = np.stack([np.maximum(fut_var, EPS),
                        np.maximum(fut_std, EPS),
                        np.maximum(fut_std, EPS),
                        np.maximum(fut_std, EPS),
                        np.maximum(fut_std, EPS)], axis=-1)   # (K, 5)
    feat_local = (feat_raw / scale).astype(np.float32)

    return feat_raw, feat_local


def query_zscore(feat: np.ndarray) -> np.ndarray:
    """(N, K, F) → per-query z-score across K dim."""
    mu  = feat.mean(axis=1, keepdims=True)
    sig = feat.std(axis=1, keepdims=True) + 1e-8
    return ((feat - mu) / sig).astype(np.float32)


def _save_features(path, raw, local_norm, zscore, n_ts, n_text):
    N, K, _ = raw.shape
    print(f'\n[save] {N} queries × {K} candidates × {N_FEATURES} features → {path}')

    print('\n── Feature statistics (raw) ──────────────────────────────────')
    for i, name in enumerate(FEATURE_NAMES):
        v = raw[:, :, i]
        print(f'  {name:20s}  mean={v.mean():.4f}  std={v.std():.4f}'
              f'  min={v.min():.4f}  max={v.max():.4f}')

    print('\n── Feature statistics (query z-score) ───────────────────────')
    for i, name in enumerate(FEATURE_NAMES):
        v = zscore[:, :, i]
        print(f'  {name:20s}  mean={v.mean():.4f}  std={v.std():.4f}')

    np.savez_compressed(
        str(path),
        feat_raw          = raw,
        feat_local_norm   = local_norm,
        feat_query_zscore = zscore,
        feature_names     = np.array(FEATURE_NAMES),
        n_ts              = np.int32(n_ts),
        n_text            = np.int32(n_text),
    )
    print(f'[save] done → {path}')
    print('Keys: feat_raw | feat_local_norm | feat_query_zscore | feature_names | n_ts | n_text')


# ─────────────────────────────────────────────────────────────────────────────
#  Path helper  (used by both Stage 1 and Stage 2 to stay in sync)
# ─────────────────────────────────────────────────────────────────────────────

def _db_subdir(args) -> str:
    """
    Returns the dataset-specific subdirectory under args.db_pred_dir, e.g.:
      pretrain  → <db_pred_dir>/pretrain/
      zeroshot  → <db_pred_dir>/ETTh1_hour_512/
                  <db_pred_dir>/ETTm1_minute_512/
                  <db_pred_dir>/electricity_hour_512/
                  …
    This guarantees Stage 1 and Stage 2 always read/write the same path.
    """
    if args.run_mode == 'pretrain':
        tag = 'pretrain'
    else:
        tag = f'{args.metadata_database_name}_{args.metadata_frequency}_{args.lookback_length}'
    return os.path.join(args.db_pred_dir, tag)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1: predict_db
# ─────────────────────────────────────────────────────────────────────────────

def stage_predict_db(args, device: str):
    """
    Pretrain:
        whole_seq (N_db, seq_len), ctx = [:ctx_len], future = [-pred_len:]
        → db_pred.npy (N_db, pred_len), db_future.npy (N_db, pred_len)

    Zeroshot:
        Saves db_rawdata.npy (n_var, n_samples); stage 2 will run bolt
        on the unique candidates it encounters (rawdata is small).
    """
    subdir = _db_subdir(args)
    os.makedirs(subdir, exist_ok=True)
    print(f'[predict_db] db subdir → {subdir}')

    if args.run_mode == 'pretrain':
        from dataset import Retriever_for_pretrain

        bolt = load_bolt(args.pretrained_model_path, device)

        retriever = Retriever_for_pretrain(
            retrieval_database_path=args.retrieval_database_path,
            dimension=768, embedding_model=None)
        retriever.build_index()

        whole_seq  = retriever.whole_seq          # (N_db, seq_len) numpy
        ctx_len    = args.retrieve_lookback_length
        pred_len   = args.prediction_length
        N_db       = len(whole_seq)
        print(f'[predict_db] pretrain  N_db={N_db}  ctx={ctx_len}  pred={pred_len}')

        ctx_all    = whole_seq[:, :ctx_len].astype(np.float32)   # (N_db, ctx_len)
        future_all = whole_seq[:, -pred_len:].astype(np.float32) # (N_db, pred_len)

        print(f'[predict_db] running ChronosBolt on {N_db} DB sequences ...')
        db_pred = bolt_predict_array(bolt, ctx_all, device, args.batch_size)

        pred_path   = os.path.join(subdir, 'db_pred.npy')
        future_path = os.path.join(subdir, 'db_future.npy')
        np.save(pred_path,   db_pred)
        np.save(future_path, future_all)
        print(f'[predict_db] saved  {pred_path}   {db_pred.shape}')
        print(f'[predict_db] saved  {future_path} {future_all.shape}')

    else:  # zeroshot
        # Persist rawdata so stage 2 can reconstruct ctx/future per candidate.
        import pickle
        from sklearn.preprocessing import StandardScaler

        db_path = (f'{args.retrieval_database_dir}{args.metadata_database_name}'
                   f'_{args.metadata_frequency}_{args.lookback_length}.pkl')
        with open(db_path, 'rb') as f:
            database = pickle.load(f)

        rawdata = np.array([database[v]['raw_data'] for v in database.keys()]).T
        scaler  = StandardScaler()
        scaler.fit(rawdata)
        rawdata = scaler.transform(rawdata).T.astype(np.float32)  # (n_var, n_samples)

        raw_path = os.path.join(subdir, 'db_rawdata.npy')
        np.save(raw_path, rawdata)
        print(f'[predict_db] zeroshot: saved rawdata {raw_path}  {rawdata.shape}')
        print('[predict_db] Stage 2 will run bolt on unique candidates encountered in queries.')


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 – pretrain
# ─────────────────────────────────────────────────────────────────────────────

def stage_compute_features_pretrain(args, device: str):
    """
    Pure index-lookup approach for pretrain:
      db_pred[i]   → ChronosBolt median prediction for DB sequence i
      db_future[i] → ground-truth future of DB sequence i
    Per query: look up TS indices (from parquet) + text indices (from npz).
    """
    import pathlib, glob as _glob
    import pandas as pd

    pred_len   = args.prediction_length
    top_k      = args.top_k
    top_k_text = args.top_k_text

    # ── Load DB predictions ───────────────────────────────────────────────
    subdir    = _db_subdir(args)
    db_pred   = np.load(os.path.join(subdir, 'db_pred.npy'))
    db_future = np.load(os.path.join(subdir, 'db_future.npy'))
    N_db = len(db_pred)
    actual_pred_len = db_pred.shape[1]
    print(f'[compute_features] pretrain  N_db={N_db}  pred_len={actual_pred_len}')

    # ── Resolve parquet files ─────────────────────────────────────────────
    data_dir = pathlib.Path(args.data_path)
    if args.file_idx is not None:
        pattern = str(data_dir / f'*chunk-{args.file_idx}*.parquet')
        parquet_files = sorted(_glob.glob(pattern))
        if not parquet_files:
            raise FileNotFoundError(f'No parquet file matches: {pattern}')
    else:
        parquet_files = sorted(str(p) for p in data_dir.glob('*.parquet'))
    print(f'[compute_features] {len(parquet_files)} parquet file(s)')

    # ── Output path ───────────────────────────────────────────────────────
    chunk_suffix = f'_chunk{args.file_idx}' if args.file_idx is not None else ''
    out_dir = pathlib.Path(args.output_dir) / 'pretrain'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'features{chunk_suffix}.npz'

    raw_list   = []
    local_list = []

    for pf in parquet_files:
        stem      = pathlib.Path(pf).stem
        chunk_str = stem.split('-chunk-')[-1] if '-chunk-' in stem else None

        # Load text KNN for this chunk (lazy, once per chunk)
        text_idx = None
        if chunk_str and args.pretrain_text_knn_dir and top_k_text > 0:
            knn_path = os.path.join(args.pretrain_text_knn_dir,
                                    f'text_knn_top20_chunk{chunk_str}.npz')
            if os.path.exists(knn_path):
                d = np.load(knn_path)
                text_idx = d['indices'][:, :top_k_text]   # (n_q, top_k_text)
                print(f'[text_knn] {knn_path}  shape={text_idx.shape}')
            else:
                print(f'[text_knn] WARNING: {knn_path} not found')

        df = pd.read_parquet(pf, columns=['indices'])
        n_rows = len(df)
        print(f'[compute_features]  chunk={chunk_str}  n_rows={n_rows}')

        for row_idx in tqdm(range(n_rows), desc=f'chunk {chunk_str}'):
            ts_indices = np.array(df.iloc[row_idx]['indices'], dtype=np.int64)[:top_k]

            if text_idx is not None and row_idx < len(text_idx):
                txt_indices = text_idx[row_idx].astype(np.int64)
            else:
                txt_indices = np.empty(0, dtype=np.int64)

            all_indices = np.concatenate([ts_indices, txt_indices])   # (K,)
            if len(all_indices) == 0:
                continue

            # Clamp to valid range
            all_indices = np.clip(all_indices, 0, N_db - 1)

            pred_k   = db_pred[all_indices]      # (K, pred_len)
            future_k = db_future[all_indices]    # (K, pred_len)

            feat_raw, feat_local = compute_features_from_lookup(pred_k, future_k)
            raw_list.append(feat_raw[np.newaxis])       # (1, K, 5)
            local_list.append(feat_local[np.newaxis])

    if not raw_list:
        raise RuntimeError('No valid rows found.')

    all_raw    = np.concatenate(raw_list,   axis=0).astype(np.float32)
    all_local  = np.concatenate(local_list, axis=0).astype(np.float32)
    all_zscore = query_zscore(all_raw)

    _save_features(out_path, all_raw, all_local, all_zscore,
                   n_ts=top_k, n_text=top_k_text)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 – zeroshot
# ─────────────────────────────────────────────────────────────────────────────

def stage_compute_features_zeroshot(args, device: str):
    """
    Zeroshot two-pass approach:
      Pass 1 – iterate loader, cache all retrieved sequences, fingerprint unique ones.
      Pass 2 – run bolt once per unique candidate ctx.
      Pass 3 – look up predictions → compute features per query.
    """
    import pathlib
    import pickle
    from sklearn.preprocessing import StandardScaler
    from data_provider.data_factory import data_provider

    ctx_len    = args.lookback_length
    pred_len   = args.pred_len
    top_k      = args.top_k
    top_k_text = args.top_k_text

    dataset_stem = args.zs_data_path.split('.')[0]
    out_dir  = pathlib.Path(args.output_dir) / 'zeroshot' / dataset_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'features.npz'

    # ── DB rawdata (for loader construction) ─────────────────────────────
    subdir   = _db_subdir(args)
    raw_path = os.path.join(subdir, 'db_rawdata.npy')
    if os.path.exists(raw_path):
        db_rawdata = np.load(raw_path)
        print(f'[compute_features] loaded db_rawdata {db_rawdata.shape}')
    else:
        db_path = (f'{args.retrieval_database_dir}{args.metadata_database_name}'
                   f'_{args.metadata_frequency}_{args.lookback_length}.pkl')
        with open(db_path, 'rb') as f:
            database = pickle.load(f)
        rawdata = np.array([database[v]['raw_data'] for v in database.keys()]).T
        scaler  = StandardScaler()
        scaler.fit(rawdata)
        db_rawdata = scaler.transform(rawdata).T.astype(np.float32)
        print(f'[compute_features] rebuilt db_rawdata {db_rawdata.shape}')

    # ── Text KNN ─────────────────────────────────────────────────────────
    text_knn_data = None
    if args.zs_text_knn_dir and top_k_text > 0:
        knn_path = os.path.join(args.zs_text_knn_dir, dataset_stem,
                                'text_knn_top20_traindb_testq.npz')
        if os.path.exists(knn_path):
            _knn = np.load(knn_path)
            text_knn_data = {'indices': _knn['indices'], 'distances': _knn['distances']}
            print(f'[text_knn] loaded {knn_path}  {text_knn_data["indices"].shape}')
        else:
            print(f'[text_knn] WARNING: {knn_path} not found — disabling text KNN')
            top_k_text = 0

    # ── Build ZS DataLoader ───────────────────────────────────────────────
    model_id = (f'{args.metadata_database_name}_zeroshot_{args.seq_len}_pred_'
                f'{pred_len}_{args.lookback_length}_retrieve_{pred_len}')
    zs_args = argparse.Namespace(
        model_id=model_id,
        embed=args.embed, percent=100, max_len=args.max_len,
        root_path=args.root_path, data_path=args.zs_data_path,
        data=args.data, features=args.features, freq=args.freq,
        target=args.target, seq_len=args.seq_len, pred_len=pred_len,
        label_len=args.label_len, batch_size=args.batch_size,
        num_workers=1, enc_in=args.enc_in,
        top_k=top_k, top_k_text=top_k_text,
        lookback_length=args.lookback_length,
        mode=args.mode, embedding_tuning=args.embedding_tuning,
        metadata={
            'lookback_length': args.lookback_length,
            'frequency':       args.metadata_frequency,
            'database_name':   [args.metadata_database_name],
        },
    )
    _, loader = data_provider(zs_args, 'test',
                              retriever_rawdata=db_rawdata,
                              text_knn_data=text_knn_data)

    total_seq_len = ctx_len + pred_len

    # ── Pass 1: collect & fingerprint unique candidates ───────────────────
    print('[compute_features] Pass 1: collecting unique DB candidates ...')
    unique_key_to_uid: dict = {}   # bytes -> int
    all_batch_uids   = []          # list of (B, K) int64 arrays
    all_batch_seqs   = []          # list of (B, K, total_seq_len) float32 arrays

    for batch in tqdm(loader, desc='pass1'):
        retrieved_seqs = batch[4]   # index-safe: works for both 6- and 7-tuple loaders
        if retrieved_seqs.dim() == 2:
            retrieved_seqs = retrieved_seqs.unsqueeze(1)
        B, K, L = retrieved_seqs.shape

        if L >= total_seq_len:
            seqs = retrieved_seqs[:, :, :total_seq_len].numpy().astype(np.float32)
        else:
            pad  = np.zeros((B, K, total_seq_len - L), dtype=np.float32)
            seqs = np.concatenate([retrieved_seqs.numpy().astype(np.float32), pad], axis=-1)

        batch_uids = np.zeros((B, K), dtype=np.int64)
        for b in range(B):
            for k in range(K):
                key = seqs[b, k].tobytes()
                if key not in unique_key_to_uid:
                    unique_key_to_uid[key] = len(unique_key_to_uid)
                batch_uids[b, k] = unique_key_to_uid[key]

        all_batch_uids.append(batch_uids)
        all_batch_seqs.append(seqs)

    n_unique  = len(unique_key_to_uid)
    n_queries = sum(arr.shape[0] for arr in all_batch_uids)
    print(f'[compute_features] unique candidates: {n_unique}  queries: {n_queries}')

    # ── Build uid_ctx / uid_future arrays ─────────────────────────────────
    uid_ctx    = np.zeros((n_unique, ctx_len),  dtype=np.float32)
    uid_future = np.zeros((n_unique, pred_len), dtype=np.float32)
    uid_done   = np.zeros(n_unique, dtype=bool)

    for seqs in all_batch_seqs:
        B, K, _ = seqs.shape
        for b in range(B):
            for k in range(K):
                uid = unique_key_to_uid[seqs[b, k].tobytes()]
                if not uid_done[uid]:
                    uid_ctx[uid]    = seqs[b, k, :ctx_len]
                    uid_future[uid] = seqs[b, k, -pred_len:]
                    uid_done[uid]   = True

    # ── Pass 2: bolt inference on unique ctx ──────────────────────────────
    bolt = load_bolt(args.pretrained_model_path, device)
    print(f'[compute_features] running bolt on {n_unique} unique candidates ...')
    uid_pred = bolt_predict_array(bolt, uid_ctx, device, args.batch_size)

    # ── Pass 3: assemble features per query ───────────────────────────────
    print('[compute_features] assembling features per query ...')
    raw_list   = []
    local_list = []

    for batch_idx, uids in enumerate(all_batch_uids):
        B = uids.shape[0]
        for b in range(B):
            pred_k   = uid_pred[uids[b]]         # (K, pred_len)
            future_k = uid_future[uids[b]]       # (K, pred_len)
            feat_raw, feat_local = compute_features_from_lookup(pred_k, future_k)
            raw_list.append(feat_raw[np.newaxis])
            local_list.append(feat_local[np.newaxis])

    if not raw_list:
        raise RuntimeError('No valid queries.')

    all_raw    = np.concatenate(raw_list,   axis=0).astype(np.float32)
    all_local  = np.concatenate(local_list, axis=0).astype(np.float32)
    all_zscore = query_zscore(all_raw)

    _save_features(out_path, all_raw, all_local, all_zscore,
                   n_ts=top_k, n_text=top_k_text)


# ─────────────────────────────────────────────────────────────────────────────
#  Argument parsing & entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Pipeline ─────────────────────────────────────────────────────────
    parser.add_argument('--stage', type=str, required=True,
                        choices=['predict_db', 'compute_features'],
                        help='Stage 1: predict all DB seqs once; '
                             'Stage 2: compute per-query features via index lookup')
    parser.add_argument('--run_mode', type=str, default='zeroshot',
                        choices=['pretrain', 'zeroshot'])

    # ── Paths ─────────────────────────────────────────────────────────────
    parser.add_argument('--db_pred_dir', type=str,
                        default='self_error_features/db/',
                        help='Stage 1 saves / Stage 2 reads db_pred.npy + db_future.npy (pretrain) '
                             'or db_rawdata.npy (zeroshot) from this directory')
    parser.add_argument('--output_dir', type=str, default='self_error_features',
                        help='Stage 2 base output directory')

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument('--pretrained_model_path', type=str,
                        default='./TS-RAG/checkpoints/base/')
    parser.add_argument('--gpu_loc',    type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)

    # ── Seq lengths ───────────────────────────────────────────────────────
    parser.add_argument('--context_length',    type=int, default=512)
    parser.add_argument('--prediction_length', type=int, default=64)

    # ── KNN ───────────────────────────────────────────────────────────────
    parser.add_argument('--top_k',      type=int, default=20)
    parser.add_argument('--top_k_text', type=int, default=20)

    # ── Pretrain ──────────────────────────────────────────────────────────
    parser.add_argument('--retrieval_database_path', type=str,
                        default='retrieval_database/pretrain/retrieval_database_512.parquet')
    parser.add_argument('--data_path', type=str,
                        default='datasets/pretrain/pretrain_pairs_ctx512')
    parser.add_argument('--retrieve_lookback_length', type=int, default=512)
    parser.add_argument('--file_idx',  type=str, default="002",
                        help='Restrict to single chunk (e.g. "000"); '
                             'output named features_chunk<idx>.npz')
    parser.add_argument('--pretrain_text_knn_dir', type=str,
                        default='Embeddings/pretrain_query')

    # ── Zeroshot ──────────────────────────────────────────────────────────
    parser.add_argument('--retrieval_database_dir', type=str,
                        default='retrieval_database/')
    parser.add_argument('--metadata_database_name', type=str, default='electricity')
    parser.add_argument('--metadata_frequency',     type=str, default='hour')
    parser.add_argument('--lookback_length',         type=int, default=512)
    parser.add_argument('--root_path',    type=str, default='datasets/electricity/')
    parser.add_argument('--zs_data_path', type=str,
                        default='electricity_retrieve_electricity_512_only_self_train_None.csv')
    parser.add_argument('--data',         type=str, default='custom_retrieve')
    parser.add_argument('--features',     type=str, default='M')
    parser.add_argument('--freq',         type=str, default='h')
    parser.add_argument('--target',       type=str, default='OT')
    parser.add_argument('--seq_len',      type=int, default=512)
    parser.add_argument('--pred_len',     type=int, default=64)
    parser.add_argument('--label_len',    type=int, default=0)
    parser.add_argument('--enc_in',       type=int, default=862)
    parser.add_argument('--embed',        type=str, default='timeF')
    parser.add_argument('--max_len',      type=int, default=-1)
    parser.add_argument('--mode',         type=str, default='only_self_train')
    parser.add_argument('--embedding_tuning', type=str, default=None)
    parser.add_argument('--zs_text_knn_dir', type=str,
                        default='Embeddings/zeroshot_query')

    args   = parser.parse_args()
    device = f'cuda:{args.gpu_loc}'

    print(f'=== gen_self_error_features  stage={args.stage}  run_mode={args.run_mode} ===')

    if args.stage == 'predict_db':
        stage_predict_db(args, device)
    else:
        if args.run_mode == 'pretrain':
            stage_compute_features_pretrain(args, device)
        else:
            stage_compute_features_zeroshot(args, device)
