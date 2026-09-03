#!/usr/bin/env python3
"""Serve QwenPI_v3 with inference-time ΠGDM RTC enabled.

The checkpoint and all existing StarVLA sources remain unchanged.  This
adapter installs the already-implemented LayerwiseFM RTC sampler as a runtime
method on QwenPI_v3 before constructing the normal policy server wrapper.
"""

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


def _install_qwenpi_v3_rtc_method() -> type:
    import numpy as np
    import torch

    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
    from starVLA.training.trainer_utils.trainer_tools import resize_images

    @torch.no_grad()
    def predict_action_realtime(
        self,
        examples=None,
        prev_action_chunk_normalized=None,
        inference_delay: int = 1,
        **kwargs,
    ) -> dict:
        """QwenPI_v3 observation path plus LayerwiseFM ΠGDM sampling."""

        if prev_action_chunk_normalized is None or inference_delay <= 0:
            return self.predict_action(examples)
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = (
            [example["state"] for example in examples]
            if "state" in examples[0]
            else None
        )

        # Match QwenPI_v3 training/inference exactly: state is quantized into
        # the instruction and is not sent through the action head state MLP.
        if state is not None:
            instructions = self.add_discretized_state_to_instruction(
                instructions, state
            )

        train_size = getattr(
            self.config.datasets.vla_data, "obs_image_size", None
        )
        if train_size:
            batch_images = resize_images(batch_images, target_size=train_size)

        vl_embs_list, _ = self._encode_vl_hidden_states(
            batch_images, instructions
        )
        device = vl_embs_list[-1].device
        previous = torch.from_numpy(
            np.asarray(prev_action_chunk_normalized, dtype=np.float32)
        ).to(device=device, dtype=torch.float32)

        # no_grad outside + enable_grad inside the action head permits the VJP
        # required by ΠGDM without retaining the expensive VLM graph.
        with torch.autocast("cuda", dtype=torch.float32):
            predicted = self.action_model.predict_action_realtime(
                vl_embs_list,
                None,
                prev_action_chunk=previous,
                inference_delay=int(inference_delay),
                **kwargs,
            )
        return {"normalized_actions": predicted.detach().cpu().numpy()}

    Qwen_PI_v3.predict_action_realtime = predict_action_realtime
    return Qwen_PI_v3


def main(args: argparse.Namespace) -> None:
    qwenpi_v3_type = _install_qwenpi_v3_rtc_method()

    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.websocket_policy_server import (
        WebsocketPolicyServer,
    )
    from examples.realRobots.Piper.eval_files.piper_eval_recording import (
        make_policy_server,
    )

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        config_overrides=args.config_override,
    )
    if not isinstance(wrapper._framework, qwenpi_v3_type):
        raise TypeError(
            "This adapter is only for QwenPI_v3; loaded "
            f"{type(wrapper._framework).__name__}"
        )
    if not wrapper.metadata.get("supports_inference_time_rtc", False):
        raise RuntimeError("QwenPI_v3 RTC method was not detected by the wrapper")

    hostname = socket.gethostname()
    logging.warning(
        "[RTC SERVER] host=%s ckpt=%s metadata=%s",
        hostname,
        args.ckpt_path,
        wrapper.metadata,
    )
    server = make_policy_server(
        WebsocketPolicyServer,
        args=args,
        server_name="qwenpi-v3-rtc",
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
