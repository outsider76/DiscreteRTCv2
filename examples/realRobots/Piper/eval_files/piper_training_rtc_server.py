#!/usr/bin/env python3
"""Serve a QwenPI_v3 TrainingRTC checkpoint for hard-prefix RTC inference."""

from __future__ import annotations

import argparse
import logging
import socket

from examples.realRobots.Piper.eval_files.piper_eval_recording import (
    add_server_recording_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=-1)
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    add_server_recording_args(parser)
    return parser


def main(args: argparse.Namespace) -> None:
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.websocket_policy_server import (
        WebsocketPolicyServer,
    )
    from examples.realRobots.Piper.eval_files.piper_eval_recording import (
        make_policy_server,
    )
    from starVLA.model.framework.VLM4A.QwenPI_v3_TrainingRTC import (
        Qwen_PI_v3_TrainingRTC,
    )

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        config_overrides=args.config_override,
    )
    if not isinstance(wrapper._framework, Qwen_PI_v3_TrainingRTC):
        raise TypeError(
            "TrainingRTC server requires framework.name="
            "'QwenPI_v3_TrainingRTC'; loaded "
            f"{type(wrapper._framework).__name__}"
        )

    action_config = wrapper._model_cfg["framework"]["action_model"]
    max_delay = int(action_config["rtc_max_delay_steps"])
    denoising_steps = int(action_config["num_inference_timesteps"])
    if max_delay <= 0:
        raise ValueError(f"Invalid rtc_max_delay_steps={max_delay}")

    metadata = dict(wrapper.metadata)
    metadata.update(
        {
            "rtc_mode": "training_time_hard_prefix",
            "rtc_max_delay_steps": max_delay,
            "rtc_delay_sampling": action_config.get(
                "rtc_delay_sampling", "uniform"
            ),
            "rtc_num_inference_timesteps": denoising_steps,
        }
    )

    logging.warning(
        "[TRAINING RTC SERVER] host=%s ckpt=%s max_delay=%d steps metadata=%s",
        socket.gethostname(),
        args.ckpt_path,
        max_delay,
        metadata,
    )
    server = make_policy_server(
        WebsocketPolicyServer,
        args=args,
        server_name="qwenpi-v3-training-rtc",
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_parser().parse_args())
