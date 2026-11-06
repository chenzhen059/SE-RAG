run_file=pretrain.py

# model
top_k=10
retrieve_lookback_length=512
retrieval_database_path=../retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet
augment_mode=moe
context_length=512
prediction_length=64

# pretrain
data_path=../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}
train_steps=6000
evaluation_steps=2000
optimizer=adamw
lr=0.00003
weight_decay=0.01
tmax=20
drop_prob=0.0
batch_size=256
shuffle_buffer_length=10000

# ── Epoch 参考 ──────────────────────────────────────────────────────────────
# 全部 30 个文件，每文件 1,000,000 行，共 30,000,000 样本
#   1 epoch ≈ 30,000,000 / 256 ≈ 117,188 步
#   train_steps=10000 ≈ 0.085 个 epoch（仅过 8.5% 数据）
#
# 只用前 10 个文件（max_files=10），共 10,000,000 样本
#   1 epoch ≈ 10,000,000 / 256 ≈ 39,063 步
#   train_steps=10000 ≈ 0.256 个 epoch
# ────────────────────────────────────────────────────────────────────────────

# 设置 max_files=10 只用前 10 个 parquet 文件；删除该行或设为 "" 则使用全部 30 个
max_files=3

# ── 文本 KNN 配置 ─────────────────────────────────────────────────────────────
# top_k_text=0 表示不使用文本索引
# text_knn_dir 指向存放 text_knn_top{top_k_text}_chunk*.npz 的目录
top_k_text=12
text_knn_dir='../Embeddings/pretrain_query'

# ── Self-error feature 配置 ───────────────────────────────────────────────────
# 由 gen_self_error_features.py --stage compute_features --run_mode pretrain 生成
# 目录下应包含 features_chunk000.npz, features_chunk001.npz, ... 等文件
# 设为 "" 则不使用 self-error 特征
self_error_feat_dir='../self_error_features/pretrain'

model_id="data50m_${augment_mode}_top${top_k}_text${top_k_text}_cosanneal_step${train_steps}_bs${batch_size}_f${max_files}_ablation_delta_offset_014"

python $run_file \
    --model_id $model_id \
    --top_k $top_k \
    --retrieve_lookback_length $retrieve_lookback_length \
    --retrieval_database_path $retrieval_database_path \
    --augment_mode $augment_mode \
    --context_length $context_length \
    --prediction_length $prediction_length \
    --data_path $data_path \
    --train_steps $train_steps \
    --evaluation_steps $evaluation_steps \
    --optimizer $optimizer \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --tmax $tmax \
    --drop_prob $drop_prob \
    --batch_size $batch_size \
    --shuffle_buffer_length $shuffle_buffer_length \
    --max_files $max_files \
    --top_k_text $top_k_text \
    --text_knn_dir $text_knn_dir \
    --self_error_feat_dir $self_error_feat_dir \
    --freeze_chronos_bolt \
    # --oracle \
    # --use_multi_gpu
