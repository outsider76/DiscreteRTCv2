# Piper evaluation viewer 数据

本目录下的四个 client 启动脚本和三个 policy server 启动脚本现在默认将评估数据保存到：

```text
/home/tams/dRTC/DiscreteRTCv2/examples/realRobots/Piper/eval_files/viewer_data
```

也可以在启动前修改输出目录：

```bash
VIEWER_DATA_DIR=/path/to/output bash examples/realRobots/Piper/eval_files/run_piper_async_rtc_client.sh
```

## Client 文件

一次 client 运行结束（包括 `Ctrl-C` 和运行异常）后会尽可能写出三个文件：

- `*_client_viewer.json`：推荐使用；可以直接由 viewer 打开。
- `*_client.npz`：相同数值数据的压缩版本。
- `*_client_summary.json`：运行参数、chunk 元数据和文件路径。

记录内容包括真实/计划发布时间、关节反馈、policy 选中的 action、经过安全限幅后发送的 command、每个返回 chunk、推理请求/完成时间及 RTC prefix 长度。相机像素不会写入这些文件。

打开下面这个固定 viewer 页面，然后选择 `*_client_viewer.json`：

```text
/home/tams/dRTC/dRTCv2/deployment/realRobots/Piper/piper_policy_viewer.html
```

也可以同时选择同一组 `*_client.npz` 和 `*_client_summary.json`。

## Server 文件

server 启动后会立即创建：

```text
*_server-*_trace.jsonl
```

首行是 server metadata，后续每行对应一次 inference，包含 request id、推理类型、RTC 参数、server 侧时延、输入 state/图片尺寸，以及 server 返回的原始 action chunk。为避免文件过大和泄露画面，server 只保存图片 shape，不保存图片像素。

client 生成的 session id 会成为 WebSocket request id 的前缀。因此可以用 client summary 中的 `session_id` 在 server JSONL 中找到同一次运行的请求。viewer 的执行时间、command、chunk 切换等数据以 client 文件为准；server trace 用于核对 server 实际返回结果与 server 侧 latency。

启动脚本会默认启用保存。直接运行 Python 文件时，需要显式添加：

```bash
--save-viewer-data --viewer-data-dir /home/tams/dRTC/DiscreteRTCv2/examples/realRobots/Piper/eval_files/viewer_data
```
