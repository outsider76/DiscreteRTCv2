#!/usr/bin/env python3
"""Serve a standard policy while recording evaluation inference traces."""

from __future__ import annotations

import argparse
import logging
import socket

from examples.realRobots.Piper.eval_files.piper_eval_recording import (
    add_server_recording_args,
    make_policy_server,
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
    parser.add_argument(
        "--binary-gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project decoded gripper output to 0/1 at --gripper-threshold (default: enabled).",
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.3)
    add_server_recording_args(parser)
    return parser


def main(args: argparse.Namespace) -> None:
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.websocket_policy_server import (
        WebsocketPolicyServer,
    )

    gripper_constraint = None
    if args.binary_gripper:
        if not 0.0 < args.gripper_threshold < 1.0:
            raise ValueError("--gripper-threshold must lie strictly between 0 and 1")
        gripper_constraint = {
            "mode": "binary",
            "dimension": -1,
            "minimum": 0.0,
            "maximum": 1.0,
            "threshold": args.gripper_threshold,
        }
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        config_overrides=args.config_override,
        gripper_constraint=gripper_constraint,
    )
    logging.warning(
        "[POLICY SERVER] host=%s ckpt=%s metadata=%s",
        socket.gethostname(),
        args.ckpt_path,
        wrapper.metadata,
    )
    server = make_policy_server(
        WebsocketPolicyServer,
        args=args,
        server_name="qwenpi-v3",
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_parser().parse_args())
