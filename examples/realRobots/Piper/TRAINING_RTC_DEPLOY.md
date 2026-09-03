# Piper TrainingRTC 部署与启动

本部署使用：

```text
results/Checkpoints/piper_pick_white_block_20260818_qwenpi_training_rtc_d15_50hz_h50/checkpoints/steps_4000_pytorch_model.pt
```

模型配置为 50 Hz、50-step action chunk，训练过的 hard-prefix delay 范围为
0–15 steps（0–300 ms）。该模型从 d10 的 step 8000 checkpoint 继续微调了
4000 steps。训练配置记录了 5 次 flow denoising；本机部署脚本默认覆盖为 4 次，
以降低异步推理延迟。去噪次数不是学习参数，这个覆盖不会改变 checkpoint。

## 1. 启动相机、Piper 和初始位置

终端 A：

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/start_piper_camera_stack.sh
```

这个脚本不启动 GELLO，并将 Piper 移动到示教起始分布附近、打开夹爪。确认
工作区清空并准备好急停。执行 VLA 时不能同时存在 GELLO 或其他
`/control/joint_states` publisher。

## 2. 启动 TrainingRTC policy server

终端 B：

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_policy_server_training_rtc.sh
```

握手 metadata 必须显示：

```text
rtc_mode: training_time_hard_prefix
rtc_max_delay_steps: 15
action_chunk_size: 50
rtc_num_inference_timesteps: 4
```

若要选择其他 checkpoint：

```bash
CKPT=/absolute/path/to/checkpoint.pt \
bash examples/realRobots/Piper/eval_files/run_policy_server_training_rtc.sh
```

如果只做离线对照、明确希望恢复 checkpoint 中的 5 次去噪，可以设置
`RTC_INFERENCE_STEPS=5`；50 Hz 真机部署不建议这样做：

```bash
RTC_INFERENCE_STEPS=5 \
bash examples/realRobots/Piper/eval_files/run_policy_server_training_rtc.sh
```

此前真机运行中，4-step TrainingRTC 请求约为 0.107–0.295 秒，客户端观测到的
delay 主要为 10–11 步、峰值 14 步；d15 覆盖该观测范围。运行真机时仍以
客户端打印的实时 `observed_delay` 为准。

## 3. 先运行 dry-run

终端 C：

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_piper_async_training_rtc_client.sh \
  --duration 30 \
  --rate-hz 50 \
  --max-joint-step 0.01 \
  --gripper-threshold 0.5 \
  --log-every 10
```

客户端默认使用 `--initial-delay-steps 15`，即训练范围的最大值，避免把 server
第一次 cold-start inference 的一次性开销当成后续异步延迟，同时确保第一次
异步请求也使用保守 prefix。后续请求仍会用实际 observed delay 更新估计，并
保留最近 10 次观测的最大值作为下一次 delay 估计。

dry-run 不发布机器人命令。重点检查：

- 后续 `observed_delay` 通常不超过 15；若频繁超过 15，应先停止真机执行并排查延迟；
- metadata 的 checkpoint、图像尺寸、action horizon 正确；
- 没有 stale camera/feedback、action starvation 或其他 controller publisher；
- chunk 切换附近的 predicted joint 没有明显跳变。

这里发送的是 `mode="simulated_delay"`，即本次重新训练的 hard-prefix
TrainingRTC。原来的 `run_piper_async_rtc_client.sh` 固定发送 `pigdm`，不能用来
验证 training-time RTC。

## 4. 短时间真机执行

完成 dry-run 后：

```bash
bash examples/realRobots/Piper/eval_files/run_piper_async_training_rtc_client.sh \
  --execute \
  --duration 10 \
  --rate-hz 50 \
  --max-joint-step 0.01 \
  --gripper-threshold 0.5 \
  --log-every 5
```

程序会再次要求输入大写 `EXECUTE`。建议第一次保持急停在手、低速短时间
测试。`--max-joint-step 0.01` 是每个 50 Hz command 相对当前反馈允许的最大
单关节变化，不是 URDF joint limit。`--gripper-threshold 0.5` 表示模型夹爪输出
大于 50% 时完全闭合，否则完全打开；Training RTC 客户端也将 0.5 设为了默认值。

## 5. 关键运行逻辑

控制线程持续以 50 Hz 消费当前 chunk；后台线程在 chunk 尚未耗尽时开始下一
次 inference。新 chunk 的前 `d` 步固定为上一 chunk 中时间戳对齐的动作，
只生成 postfix。新结果到达后，控制线程按它的原始时间戳索引继续执行，不会
从新 chunk 的 step 0 重新开始。

如果保守 delay 估计超过训练上限 15，当前客户端会把送入模型的 conditioning
delay 截断为 15，并打印 `TrainingRTC delay clamp` 警告。这样不会仅因为
偶发的 `d=16` 就关闭 external control gate，但实际到达时间超过第 15 步的部分不再是
hard-prefix，而是模型生成的 postfix，因此 chunk 边界连续性没有训练保证。

这个 clamp 是临时降级策略，不等同于模型支持更长延迟。如果运行中经常出现
16 steps 或更高，应先检查 GPU 竞争、相机/ROS 调度和控制循环 deadline miss，
再决定是否扩大训练 delay；更长随机 prefix 会减少每个样本用于 postfix loss 的
平均长度。
