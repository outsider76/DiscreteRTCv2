# Piper QwenPI training-time RTC 微调

本文对应：

- 数据：`data/piper_lerobot/20260818_piper_pick_white_block_50hz`
- 基础任务 checkpoint：`results/Checkpoints/piper_pick_white_block_20260818_qwenpi_measured_50hz_h50/checkpoints/steps_10000_pytorch_model.pt`
- 论文：`/home/tams/Desktop/2512.05964v2.pdf`
- 参考实现：`/home/tams/real-time-chunking-kinetix/src/model.py`

## 1. 为什么需要新的训练实现

原来的 `QwenPI_v3 + LayerwiseFM` 给整个 action chunk 使用同一个 flow
time。training-time RTC 要求一个 chunk 中每个 action token 可以有不同的
time：已知 prefix 使用 `t=1`，待生成 postfix 使用随机 flow time `t`。原
DiT 的 AdaLN 也只接收每个 batch 一个 time，因此不能只在数据层加 mask。

本实现新增两个文件，没有修改原训练 head/framework：

- `starVLA/model/modules/action_model/TrainingRTC_LayerwiseFM_ActionHeader.py`
- `starVLA/model/framework/VLM4A/QwenPI_v3_TrainingRTC.py`

新 head 复用原 LayerwiseFM 的所有可学习层，参数名称和 shape 完全一致，
所以能够完整加载已经完成任务学习的 QwenPI_v3 checkpoint，再进行 RTC
适配训练。

## 2. 训练 loss

对每个训练样本独立采样 delay `d`：

```text
d ~ Uniform{0, 1, ..., 10}
prefix  = action[0:d]
postfix = action[d:50]
```

prefix 使用干净的 ground-truth action，并将其 token flow time 设为 1；
postfix 使用标准 flow-matching 加噪。模型同时看到两部分，但 loss 只计算
postfix：

```text
x_t[i] = action[i]                         i < d
x_t[i] = t * action[i] + (1-t) * noise[i] i >= d

loss = sum(((v_pred - (action-noise))^2) * postfix_mask)
       / sum(postfix_mask)
```

推理时，每个 Euler 去噪步都会重新固定 prefix，且 prefix token 继续使用
`t=1`，因此不会被去噪网络改写。

## 3. 本次训练 setup

| 项目 | 设置 |
|---|---:|
| 数据频率 | 50 Hz |
| action/state | 7 维：6 joints + 1 gripper |
| 图像 | global + hand，两路，224×224 |
| action horizon | 50 steps（1 秒） |
| RTC delay 训练范围 | 0–10 steps（0–200 ms，含端点） |
| delay 分布 | uniform |
| flow inference steps | 5 |
| optimizer steps | 8000 |
| micro batch | 1 |
| gradient accumulation | 8 |
| repeated flow samples | 2 |
| frozen module | Qwen3-VL backbone |
| action-head LR | 5e-5 |
| projector LR | 2e-5 |
| warmup | 200 steps |
| checkpoint interval | 2000 steps |

论文的真实机器人实验同样使用 50 Hz、最大 10-step delay、5 次去噪和
8000 次更新，但 batch size 是 512。本机单卡设置采用有效 batch 8，并在
每个样本上重复 2 次 flow noise；因此实现机制一致，但并非对论文算力配置
的逐项复现。

训练使用 measured Piper joint targets。数据注册项
`piper_pick_white_block_20260818_measured_50hz_h50` 的实际单样本结构已经
检查为：两路图像、`state=(1, 7)`、`action=(50, 7)`。

## 4. 启动和查看训练

训练配置：

```text
examples/realRobots/Piper/train_files/starvla_qwenpi_piper_20260818_training_rtc_50hz_h50.yaml
```

启动：

```bash
cd /home/tams/DiscreteRTCv2
mkdir -p Log
bash examples/realRobots/Piper/train_files/run_piper_qwenpi_20260818_training_rtc_train.sh \
  2>&1 | tee Log/piper_qwenpi_20260818_training_rtc_d10_50hz_h50.log
```

查看：

```bash
tail -f Log/piper_qwenpi_20260818_training_rtc_d10_50hz_h50.log
```

输出目录：

```text
results/Checkpoints/piper_pick_white_block_20260818_qwenpi_training_rtc_d10_50hz_h50/
```

正常情况下会保存 `steps_2000`、`steps_4000`、`steps_6000`、
`steps_8000` 和 final model。每一个 `steps_N` 都是当时模型参数的完整快照，
不是额外的 action chunk。

## 5. 运行时约束

这个模型学到的是 hard-prefix conditioning。异步部署时应调用
`predict_action_realtime(..., mode="simulated_delay")`，并提供上一 chunk 中
与新 chunk 时间戳重叠的 normalized action prefix。实际 delay 必须不大于
训练上限 10 steps；超过 200 ms 时应降级、等待或停止，不应把超范围 prefix
直接交给模型。

training-time RTC 不需要 ΠGDM 的 VJP，因此每个 denoising step 只需一次模型
前向；它与未训练模型所用的 inference-time RTC/ΠGDM 是两条不同路径。
