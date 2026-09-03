# Piper 异步 Action Chunk 控制

本文档说明两个新增、彼此独立的异步控制方法：

1. inference-time Real-Time Chunking（RTC，ΠGDM soft-mask guidance）；
2. 基于全局控制时间戳的双 action-chunk temporal ensembling（TE）。

现有同步客户端、训练配置、checkpoint 和通用 policy server 均未修改。
RTC 只使用 inference-time 方法，不包含论文中的 simulated-delay training；当前
QwenPI_v3 checkpoint 不需要重新训练。

## 新增文件

- `eval_files/piper_async_rtc_client.py`：RTC 异步控制入口；
- `eval_files/piper_async_temporal_ensemble_client.py`：TE 异步控制入口；
- `eval_files/piper_async_common.py`：两个入口共用的 ROS、安全检查、50 Hz
  控制线程和后台推理线程；
- `eval_files/piper_qwenpi_v3_rtc_server.py`：给 QwenPI_v3 动态暴露已有的
  LayerwiseFM ΠGDM sampler；
- `eval_files/run_policy_server_rtc.sh`：RTC policy server 启动脚本；
- `eval_files/run_piper_async_rtc_client.sh`：RTC 客户端环境启动脚本；
- `eval_files/run_piper_async_temporal_ensemble_client.sh`：TE 客户端环境启动脚本。

## 公共执行结构

模型的 action horizon 为 `H=50`，控制频率为 `50 Hz`，所以一个 chunk
覆盖一秒。首次 action chunk 会在打开真机控制 gate 之前同步生成，用来保证
控制开始时已有可执行 action；真机开始运动后，控制线程不再等待网络或模型：

```text
ROS callbacks ──> 最新相机/关节 snapshot ──> 后台 inference worker
                                             │
                                             v
50 Hz control loop <── timestamped chunks <── policy server
        │
        └── joint step limit + binary gripper ──> /control/joint_states
```

每个推理请求记录发起时的整数控制 step，返回 chunk 的第 0 行与该 step 对齐。
因此，即使推理花费多个控制周期，客户端也不会错误地从新 chunk 的第 0 行开始
执行，而是使用与当前控制时间戳对应的行。

## 方法一：inference-time RTC

参考：

- 本地论文 `/home/tams/Desktop/2506.07339v2.pdf`，Real-Time Execution of
  Action Chunking Flow Policies，尤其是第 3 节和 Algorithm 1；
- `/home/tams/real-time-chunking-kinetix/src/model.py` 中的
  `FlowPolicy.realtime_action`；
- `/home/tams/real-time-chunking-kinetix/src/eval_flow.py` 中的异步 chunk
  对齐方式。

### 客户端调度

RTC 默认使用论文实机参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `H` | 50 | checkpoint 给出的 action horizon |
| `s_min` | 25 steps | `--min-execution-horizon`，最小执行区间 |
| `beta` | 5 | `--max-guidance-weight`，ΠGDM guidance clip |
| delay buffer | 10 | `--delay-buffer-size` |
| soft-mask schedule | `exp` | 指数衰减，与论文默认一致 |

假设当前 chunk 的 origin 为 `t0`，控制器执行到全局 step `t`：

1. 计算 `s = t - t0`；当 `s >= max(s_min, d)` 时启动后台推理；
2. 将旧 chunk 的 `A_old[s:H]` 作为与新 observation 对齐的剩余前缀；
3. 用最近 inference delay buffer 的最大值作为保守预测 `d`；
4. server 在每个 flow denoising step 使用 ΠGDM VJP guidance：前 `d` 行权重
   为 1，中间重叠区指数衰减，最后 `s` 行权重为 0；
5. 推理期间控制器继续执行旧 chunk；
6. 假如结果在 `d_actual` 个控制 step 后返回，新 chunk 的 origin 仍是请求发起
   step，下一条命令直接取 `A_new[d_actual]`，不会重放已经过去的行；
7. 将 `d_actual` 写入 delay buffer，供下一次保守预测使用。

使用的是 `mode=pigdm`，不会启用需要 simulated-delay 重新训练的
`mode=simulated_delay`。

RTC 必须满足旧 chunk 在推理完成前仍有 action，即论文约束
`d <= s <= H-d`。若实测 delay 超过 `H/2`、结果返回时整个 chunk 已过期，或
没有可执行 action，客户端会关闭 control gate 并报错，不会用未知动作继续运动。

### 为什么需要新的 RTC server adapter

当前 LayerwiseFM action head、normalization wrapper 和 WebSocket router 已经包含
RTC sampler/API，但 QwenPI_v3 framework 没有公开 `predict_action_realtime`，普通
server 的 metadata 因此会显示：

```text
supports_inference_time_rtc: false
```

新增 adapter 只在进程运行时给 QwenPI_v3 绑定该方法，并复用原有 checkpoint、
state-to-language token、图像预处理、训练期 action normalization 和 ΠGDM action
head。它不会写回或修改任何模型源文件。RTC server 启动后 metadata 应显示：

```text
supports_inference_time_rtc: true
```

### 启动 RTC

终端 1：启动相机和 Piper，不能启动 GELLO 或其他
`/control/joint_states` publisher。

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/start_piper_camera_stack.sh
```

终端 2：启动 RTC server。

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_policy_server_rtc.sh
```

终端 3：先 dry-run。`--log-every 5` 只降低打印频率，不改变 50 Hz 控制时序。

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_piper_async_rtc_client.sh \
  --duration 10 \
  --rate-hz 50 \
  --min-execution-horizon 25 \
  --max-joint-step 0.01 \
  --log-every 5
```

确认 dry-run 中持续出现 `[ASYNC RTC START]` 和 `[ASYNC INFER READY]`，且
`observed_delay < 25` 后，再进行短时间真机测试：

```bash
bash examples/realRobots/Piper/eval_files/run_piper_async_rtc_client.sh \
  --execute \
  --duration 5 \
  --rate-hz 50 \
  --min-execution-horizon 25 \
  --max-joint-step 0.01 \
  --log-every 5
```

## 方法二：双 chunk temporal ensembling

TE 使用普通 flow inference，不需要 RTC sampler。可以连接已有的
`run_policy_server.sh`，也可以连接新增 RTC server，因为 RTC server 同时保留普通
`infer` endpoint。

后台 worker 尽可能密集地推理。`--inference-interval-steps 1` 表示上一次请求
完成后，只要距离上次请求发起至少一个控制 step，就立即发起下一次请求；同一时刻
仍然只有一个在途请求，不会并发占用多份 GPU memory。

只保留最新两个 chunk。旧、新 chunk 的 origin 分别为 `t_old`、`t_new`，对全局
控制 step `t`，若两者都覆盖该 step，则：

```text
i_old = t - t_old
i_new = t - t_new
a(t) = (1 - w_new) * A_old[i_old] + w_new * A_new[i_new]
```

默认 `w_new=0.5`。例如 `--new-chunk-weight 0.35` 会给旧预测 0.65、新预测
0.35，更偏向跨 chunk 平滑；提高 `w_new` 会更快响应最新 observation。如果只有
一个 chunk 覆盖当前 step，则直接使用该 chunk。平均的是模型的原始 7-D 输出；
之后 arm joints 经过 joint-step limit，夹爪平均值再按 0.5 阈值转换成全开/全闭。

论文将 TE 作为 baseline，并指出在多模态任务里“两个都有效的动作的平均”不一定
仍然是有效动作。因此 TE 适合做对照实验，不能默认认为它一定比 RTC 更稳定。

### 启动 TE

终端 1 启动相机和 Piper；终端 2 启动普通 server：

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_policy_server.sh
```

终端 3 先 dry-run：

```bash
cd /home/tams/DiscreteRTCv2
bash examples/realRobots/Piper/eval_files/run_piper_async_temporal_ensemble_client.sh \
  --duration 10 \
  --rate-hz 50 \
  --inference-interval-steps 1 \
  --new-chunk-weight 0.5 \
  --max-joint-step 0.01 \
  --log-every 5
```

短时间真机测试：

```bash
bash examples/realRobots/Piper/eval_files/run_piper_async_temporal_ensemble_client.sh \
  --execute \
  --duration 5 \
  --rate-hz 50 \
  --inference-interval-steps 1 \
  --new-chunk-weight 0.5 \
  --max-joint-step 0.01 \
  --log-every 5
```

TE 控制日志中的：

```text
chunks=2:14,3:6 w=0.500,0.500
```

表示当前全局 timestamp 同时对应 chunk 2 的第 14 行和 chunk 3 的第 6 行，输出
是二者各 0.5 的加权平均。

## 两种方法的区别

| 项目 | RTC | Temporal ensembling |
|---|---|---|
| 控制线程是否等待 inference | 否 | 否 |
| 是否修改 flow sampling | 是，ΠGDM VJP guidance | 否 |
| 是否需要重新训练 | inference-time 模式不需要 | 不需要 |
| 跨 chunk 连续性 | 在生成过程中约束整条新轨迹 | 对已生成的两个动作做平均 |
| 推理计算量 | 较高，每个 denoising step 需要反向 VJP | 与普通 inference 相同 |
| 多模态动作风险 | 较低，但仍需真机验证 | 平均可能落在两个动作模态之间 |
| 使用的 server | `run_policy_server_rtc.sh` | 普通或 RTC server 均可 |

## 仍然生效的真机保护

两个异步客户端保留同步客户端的关键保护：

- dry-run 默认不发布命令；
- `--execute` 必须在交互终端输入 `EXECUTE`；
- 检测其他 `/control/joint_states` publisher；
- robot/camera 数据过期立即停止；
- arm joint state 超出训练范围时停止，夹爪运行期不使用训练范围阻断；
- arm joints 每条命令受 `--max-joint-step` 限制；
- 夹爪保持二值逻辑：模型输出 `>0.5` 全闭，否则全开；
- duration、异常或 `Ctrl+C` 都会关闭 `/control_enable`；
- async chunk 过期或 action starvation 时停止，不会自动 hold 或重放过期 action。

这些客户端和现有同步客户端一样，没有加入 URDF joint limit 检查；
`--max-joint-step` 是相对当前反馈的单周期变化限制，不是机器人本体绝对 joint limit。

## 建议的验证顺序

1. 确认机器人处于示教起始位置、夹爪打开；
2. RTC server 启动时确认 metadata 的 RTC 字段为 `true`；
3. 分别运行 10 秒 dry-run，记录 inference latency 和 `observed_delay`；
4. RTC 必须保持 `observed_delay < 25`；如果接近 25，不要真机执行；
5. 检查日志中控制 step 连续递增，新 chunk 使用的 index 等于 timestamp 差值；
6. 首次真机只运行 5 秒，并握住急停；
7. 对比 RTC 与 TE 的 joint command 一阶差分、二阶差分和任务成功率，再决定正式参数。
