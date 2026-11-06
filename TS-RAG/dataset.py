import torch
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

from gluonts.itertools import Cyclic
from torch.utils.data import IterableDataset
from gluonts.dataset.common import FileDataset


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class CustomPretrainDataset(IterableDataset, ShuffleMixin):
    def __init__(
        self,
        dataset_path,
        retriever,
        mode="training",
        drop_prob=0.2,
        context_length=512,
        prediction_length=64,
        retrieve_lookback_length=64,
        top_k=5,
        max_files=None,          # NEW: 只读前 N 个 parquet 文件（None = 全部）
        text_knn_dir=None,       # NEW: Embeddings/pretrain_query 目录，存放 text_knn_top{k}_chunk*.npz
        top_k_text=0,            # NEW: 文本近邻数量，0=不使用
        text_db_rawdata=None,    # NEW: retriever.whole_seq（n_db, seq_len），用于还原文本检索序列
        self_error_feat_dir=None, # NEW: self_error_features/pretrain/ 目录，存放 features_chunk*.npz
    ):
        super().__init__()

        assert mode in ("training", "validation", "test")

        self.drop_prob = drop_prob
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.retriever = retriever
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.retrieve_lookback_length = retrieve_lookback_length
        self.top_k = top_k
        self.max_files = max_files
        self.text_knn_dir = Path(text_knn_dir) if text_knn_dir else None
        self.top_k_text = top_k_text
        self.text_db_rawdata = text_db_rawdata  # (n_db, seq_len) numpy array
        # 文本 KNN 数据按 chunk 延迟加载（在 __iter__ 里按需读取）
        self._text_knn_cache: dict = {}   # chunk_str -> {'indices': ..., 'distances': ...}
        # self-error features 按 chunk 延迟加载，与 text KNN 加载方式完全对称
        self.self_error_feat_dir = Path(self_error_feat_dir) if self_error_feat_dir else None
        self._self_error_feat_cache: dict = {}  # chunk_str -> np.ndarray (N_chunk, K_saved, 5)

        # Ensure the dataset path exists
        if not self.dataset_path.is_dir():
            raise ValueError(f"Provided dataset_path {dataset_path} is not a directory.")

        # 所有 parquet 文件，排序后按 max_files 截断
        all_files = sorted(
            [f for f in self.dataset_path.iterdir() if f.suffix == ".parquet"]
        )
        if not all_files:
            raise ValueError("All files in the dataset_path should be parquet files.")

        if max_files is not None:
            self.parquet_files = all_files[:max_files]
            print(f"[CustomPretrainDataset] max_files={max_files}: "
                  f"using {len(self.parquet_files)}/{len(all_files)} files")
        else:
            self.parquet_files = all_files

        # 若不限制文件数，仍走原来的 FileDataset 路径（保持兼容）；
        # 若限制了，改用自定义 parquet 迭代器（不依赖 FileDataset 的目录扫描）
        self._use_custom_iter = (max_files is not None)

        if not self._use_custom_iter:
            self.dataset = FileDataset(self.dataset_path, freq="1H")
            if self.mode == "training":
                self.dataset = Cyclic(self.dataset)


    def _load_text_knn(self, chunk_str: str):
        """按需加载 text KNN 文件，结果缓存到 self._text_knn_cache。"""
        if chunk_str in self._text_knn_cache:
            return self._text_knn_cache[chunk_str]
        if self.text_knn_dir is None or self.top_k_text <= 0:
            return None
        knn_path = self.text_knn_dir / f"text_knn_top{20}_chunk{chunk_str}.npz"
        if not knn_path.exists():
            print(f"[text_knn] WARNING: {knn_path} not found, text retrieval disabled for chunk {chunk_str}")
            self._text_knn_cache[chunk_str] = None
            return None
        data = np.load(knn_path)
        result = {'indices': data['indices'], 'distances': data['distances']}  # (n_query, top_k_text)
        self._text_knn_cache[chunk_str] = result
        return result

    def _load_self_error_feat(self, chunk_str: str):
        """按需加载 self-error feature 文件，结果缓存到 self._self_error_feat_cache。
        与 _load_text_knn 完全对称。
        文件命名规则：features_chunk{chunk_str}.npz（由 gen_self_error_features.py 生成）
        返回 np.ndarray (N_chunk, K_saved, 5) 或 None。
        """
        if chunk_str in self._self_error_feat_cache:
            return self._self_error_feat_cache[chunk_str]
        if self.self_error_feat_dir is None:
            return None
        feat_path = self.self_error_feat_dir / f"features_chunk{chunk_str}.npz"
        if not feat_path.exists():
            print(f"[self_error_feat] WARNING: {feat_path} not found, self-error feat disabled for chunk {chunk_str}")
            self._self_error_feat_cache[chunk_str] = None
            return None
        data = np.load(feat_path)
        # feat_query_zscore: (N_chunk, K_saved, 5), K_saved = n_ts_saved + n_text_saved
        arr = data['feat_query_zscore'].astype(np.float32)
        self._self_error_feat_cache[chunk_str] = {
            'feat':   arr,
            'n_ts':   int(data['n_ts']),
            'n_text': int(data['n_text']),
        }
        return self._self_error_feat_cache[chunk_str]

    def _iter_parquet_files(self):
        """直接从 self.parquet_files 读取行，无限循环（training）或单次（其他）。"""
        def _one_pass():
            for pf in self.parquet_files:
                # 从文件名解析 chunk_str（如 chunk-000 → "000"）
                stem = pf.stem  # e.g. "chronos-dataset-sample-50m-chunk-000"
                chunk_str = stem.split("-chunk-")[-1] if "-chunk-" in stem else None
                text_knn   = self._load_text_knn(chunk_str)       if chunk_str else None
                self_err   = self._load_self_error_feat(chunk_str) if chunk_str else None

                df = pd.read_parquet(pf, columns=['target', 'distances', 'indices'])
                for row_idx, (_, row) in enumerate(df.iterrows()):
                    entry = {
                        'target':    np.array(row['target'],    dtype=np.float32),
                        'distances': np.array(row['distances'], dtype=np.float32),
                        'indices':   np.array(row['indices'],   dtype=np.int64),
                    }
                    if text_knn is not None and row_idx < len(text_knn['indices']):
                        entry['text_indices']   = text_knn['indices'][row_idx].astype(np.int64)    # (top_k_text,)
                        entry['text_distances'] = text_knn['distances'][row_idx].astype(np.float32) # (top_k_text,)
                    # self-error features: 每行 (K_saved, 5)，_process_entry 中再截断
                    if self_err is not None and row_idx < len(self_err['feat']):
                        entry['self_error_feat'] = self_err['feat'][row_idx]   # (K_saved, 5)
                        entry['_self_err_n_ts']  = np.int64(self_err['n_ts'])
                        entry['_self_err_n_text']= np.int64(self_err['n_text'])
                    yield entry

        if self.mode == "training":
            while True:           # Cyclic 等价
                yield from _one_pass()
        else:
            yield from _one_pass()

    def _process_entry(self, entry):
        """公共字段裁剪逻辑，training / non-training 共用。"""
        entry['x'] = entry['target'][:self.context_length]
        entry['y'] = entry['target'][self.context_length:]
        entry['distances'] = entry['distances'][:self.top_k]
        entry['indices']   = entry['indices'][:self.top_k]
        # 截断文本 KNN 到 top_k_text（npz 文件存储的 top_k 可能更大）
        if 'text_indices' in entry:
            entry['text_indices']   = entry['text_indices'][:self.top_k_text]
            entry['text_distances'] = entry['text_distances'][:self.top_k_text]
        if 'text_indices' not in entry:
            entry['text_indices']   = np.empty(0, dtype=np.int64)
            entry['text_distances'] = np.empty(0, dtype=np.float32)
        # self-error features: 重排到 (top_k+top_k_text, 5)，TS 在前 Text 在后
        K_total = self.top_k + self.top_k_text
        if 'self_error_feat' in entry:
            raw   = entry['self_error_feat']          # (K_saved, 5)
            n_ts  = int(entry['_self_err_n_ts'])
            n_txt = int(entry['_self_err_n_text'])
            out   = np.zeros((K_total, 5), dtype=np.float32)
            ts_take  = min(self.top_k,      n_ts)
            txt_take = min(self.top_k_text, n_txt)
            if ts_take  > 0: out[:ts_take,              :] = raw[:ts_take, :]
            if txt_take > 0: out[self.top_k:self.top_k + txt_take, :] = raw[n_ts:n_ts + txt_take, :]
            entry['self_error_feat'] = out
            del entry['_self_err_n_ts'], entry['_self_err_n_text']
        else:
            entry['self_error_feat'] = np.zeros((K_total, 5), dtype=np.float32)
        return entry

    def __iter__(self):
        if self._use_custom_iter:
            # ── 自定义 parquet 迭代器（max_files 模式）────────────────────────
            if self.mode == "training":
                for entry in self._iter_parquet_files():
                    entry = self._process_entry(entry)
                    if self.drop_prob > 0:
                        target = entry['target'].copy()
                        drop_p = np.random.uniform(0.0, self.drop_prob)
                        mask = np.random.choice(
                            [True, False], size=len(target), p=[drop_p, 1 - drop_p]
                        )
                        target[mask] = np.nan
                        entry['target'] = target
                    yield entry
            else:
                for entry in self._iter_parquet_files():
                    yield self._process_entry(entry)

        else:
            # ── 原始 FileDataset 路径（全部文件）────────────────────────────
            iterable = iter(self.dataset)
            if self.mode == "training":
                while True:
                    entry = next(iterable)
                    entry = {f: entry[f] for f in ['target', 'distances', 'indices']}
                    entry = self._process_entry(entry)
                    if self.drop_prob > 0:
                        target = entry['target'].copy()
                        drop_p = np.random.uniform(0.0, self.drop_prob)
                        mask = np.random.choice(
                            [True, False], size=len(target), p=[drop_p, 1 - drop_p]
                        )
                        target[mask] = np.nan
                        entry['target'] = target
                    yield entry
            else:
                for entry in iterable:
                    entry = {f: entry[f] for f in ['target', 'distances', 'indices']}
                    yield self._process_entry(entry)


class Retriever_for_pretrain():
    def __init__(self, retrieval_database_path, dimension, embedding_model):
        self.retrieval_database_path = retrieval_database_path
        self.d = dimension #768
        self.index = None
        self.Y = None
        self.embedding_model = embedding_model

    def build_index(self):
        self.index = faiss.IndexFlatL2(self.d)  # euclidean distance

        database = pd.read_parquet(self.retrieval_database_path)
        embeddings = np.vstack(database["embedding"].to_numpy())
        self.x = database['x'].values
        self.y = database['y'].values
        self.whole_seq = np.concatenate([self.x.tolist(), self.y.tolist()], axis=-1)
        self.index.add(embeddings)

    def embedding(self, x_tensor):
        embeddings, _ = self.embedding_model.embed(x_tensor)
        return embeddings[:, -1, :].float().numpy()

    def search(self, query_vector, top_k, params=None):
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        # drop first or last
        if params is None:
            distances, indices = self.index.search(query_vector, top_k + 1)
        else:
            distances, indices = self.index.search(query_vector, top_k + 1, params=params)
        # drop first if first distance is 0
        mask = distances[:, 0] == 0
        distances = np.where(
            mask[:, None],
            distances[:, 1:], 
            distances[:, :-1]
        )
        indices = np.where(
            mask[:, None],
            indices[:, 1:], 
            indices[:, :-1]
        )
        
        return indices, distances


