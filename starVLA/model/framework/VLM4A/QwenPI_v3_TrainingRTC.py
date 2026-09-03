"""QwenPI_v3 framework for training-time Real-Time Chunking.

The vision-language path is inherited unchanged from QwenPI_v3.  Only the
LayerwiseFM action head factory is replaced during construction, preserving
all parameter names and shapes for full warm-start from a vanilla checkpoint.
"""

from __future__ import annotations

import importlib
from typing import Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
from starVLA.model.modules.action_model.TrainingRTC_LayerwiseFM_ActionHeader import (
    get_action_model as get_training_rtc_action_model,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


qwenpi_v3_module = importlib.import_module(
    "starVLA.model.framework.VLM4A.QwenPI_v3"
)


@FRAMEWORK_REGISTRY.register("QwenPI_v3_TrainingRTC")
class Qwen_PI_v3_TrainingRTC(Qwen_PI_v3):
    """QwenPI_v3 with hard-prefix action conditioning during training."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        original_factory = qwenpi_v3_module.get_action_model
        qwenpi_v3_module.get_action_model = get_training_rtc_action_model
        try:
            super().__init__(config=config, **kwargs)
        finally:
            qwenpi_v3_module.get_action_model = original_factory

    @torch.inference_mode()
    def predict_action_realtime(
        self,
        examples=None,
        prev_action_chunk_normalized=None,
        inference_delay: int = 1,
        mode: str = "simulated_delay",
        **kwargs,
    ) -> dict:
        """Generate with the hard prefix learned by training-time RTC."""

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
        if state is not None:
            instructions = self.add_discretized_state_to_instruction(
                instructions, state
            )

        training_image_size = getattr(
            self.config.datasets.vla_data, "obs_image_size", None
        )
        if training_image_size:
            batch_images = resize_images(
                batch_images, target_size=training_image_size
            )

        vl_embs_list, attention_mask = self._encode_vl_hidden_states(
            batch_images, instructions
        )
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        device = vl_embs_list[-1].device
        previous = torch.from_numpy(
            np.asarray(prev_action_chunk_normalized, dtype=np.float32)
        ).to(device=device, dtype=torch.float32)

        with torch.autocast("cuda", dtype=torch.float32):
            predicted = self.action_model.predict_action_realtime(
                vl_embs_list,
                None,
                prev_action_chunk=previous,
                inference_delay=int(inference_delay),
                mode=mode,
                encoder_attention_mask=attention_mask,
                **kwargs,
            )
        return {"normalized_actions": predicted.detach().cpu().numpy()}
