# UniGSA

## 依赖安装

| 库 | 版本 |
| --- | --- |
| PyTorch | 2.11.0 |
| Transformers | 4.57.1 |
| PEFT | 0.18.1 |
| Accelerate | 1.12.0 |
| BeautifulSoup4 | 4.14.2 |

## 运行方式

### 准备路径

以下命令在 `UniGSA` 文件夹内执行，将模型和数据路径替换为实际路径：

```bash
cd /path/to/UniGSA
export MODEL=/path/to/Qwen3-8B
export DATA_ROOT=/path/to/TTQA_dataset
```

`MODEL` 指向包含模型权重、配置和 tokenizer 的本地预训练模型目录。数据目录使用以下结构：

```text
TTQA_dataset/
├── hitab/
│   ├── tables.jsonl
│   ├── train_samples_processed.jsonl
│   ├── dev_samples_processed.jsonl
│   └── test_samples_processed.jsonl
├── FinQA/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── TAT-QA/
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── MultiHiertt/
    ├── train.json
    ├── dev.json
    └── test.json
```

### 训练

HiTab、FinQA、TAT-QA 使用 `train_and_eval_v2.py`，通过以下脚本启动单卡训练：

```bash
# HiTab
GPUS=0 DATA="$DATA_ROOT/hitab" bash scripts/train_hitab_qwen3.sh

# FinQA
GPUS=0 DATA="$DATA_ROOT/FinQA/train.json" bash scripts/train_finqa_qwen3.sh

# TAT-QA
GPUS=0 DATA="$DATA_ROOT/TAT-QA/train.json" bash scripts/train_tatqa_qwen3.sh
```

MultiHiertt 使用 `train_and_eval_mp.py`，单进程将模型分布到多张 GPU，直接启动脚本即可，不使用 `torchrun`：

```bash
GPUS=0,1,2,3 DATA="$DATA_ROOT/MultiHiertt/train.json" \
  bash scripts/train_multihiertt_qwen3_mp.sh
```

训练脚本每个 epoch 后评估验证集并保存检查点，最终模型保存到 `output/<数据集>/final`。可通过环境变量 `OUTPUT` 设置输出目录、`SEED` 设置随机种子，在命令末尾追加参数覆盖默认配置：

```bash
GPUS=0,1 DATA="$DATA_ROOT/MultiHiertt/train.json" \
  OUTPUT=./output/multihiertt_custom SEED=43 \
  bash scripts/train_multihiertt_qwen3_mp.sh \
    --max_length 8192 --num_epochs 3 --grad_accum 4 \
    --eval_data_path "$DATA_ROOT/MultiHiertt/dev.json"
```

默认训练 3 个 epoch，batch size 为 1，梯度累积为 2。HiTab 学习率为 `3e-5`，其余为 `5e-5`；MultiHiertt 最大长度为 4096，其余为 2048。超长输入从右侧截断；若训练答案被全部截掉，该样例会提示并跳过。

依次运行种子 42、43、44：

```bash
GPUS=0 DATA="$DATA_ROOT/FinQA/train.json" \
  bash scripts/run_three_seeds.sh finqa
```

数据集参数可选 `hitab`、`finqa`、`tatqa`、`multihiertt`，结果分别保存到 `output/<数据集>/seed_<种子>/final`。

### 评估

评估使用与训练相同的 `MODEL`，`CHECKPOINT` 指向新训练的 UniGSA 检查点目录。以下命令使用单张 GPU：

```bash
# HiTab
CUDA_VISIBLE_DEVICES=0 CHECKPOINT=./output/hitab/final \
  DATA="$DATA_ROOT/hitab" bash scripts/eval_hitab.sh

# FinQA
CUDA_VISIBLE_DEVICES=0 CHECKPOINT=./output/finqa/final \
  DATA="$DATA_ROOT/FinQA/test.json" bash scripts/eval_finqa.sh

# TAT-QA
CUDA_VISIBLE_DEVICES=0 CHECKPOINT=./output/tatqa/final \
  DATA="$DATA_ROOT/TAT-QA/test.json" bash scripts/eval_tatqa.sh

# MultiHiertt
CUDA_VISIBLE_DEVICES=0 CHECKPOINT=./output/multihiertt/final \
  DATA="$DATA_ROOT/MultiHiertt/test.json" bash scripts/eval_multihiertt.sh
```

预测结果默认保存到 `output/<数据集>_predictions.json`，可用环境变量 `OUTPUT` 指定输出文件。评估脚本可追加 `--fp16` 使用半精度加载，或追加 `--max_length 8192` 调整输入长度。

### 查看完整参数

直接使用 Python 模块入口时，在 `UniGSA` 的父目录执行：

```bash
cd ..
conda run -n xq_py311 python -m UniGSA.train_and_eval_v2 --help
conda run -n xq_py311 python -m UniGSA.train_and_eval_mp --help
conda run -n xq_py311 python -m UniGSA.inference_cli --help
```
