from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """

        image = image.convert('RGB')
        size = self.config.image_size
        image = image.resize((size, size))

        arr = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        tensor = (tensor - mean) / std

        num_tiles = self.config.num_tiles

        if num_tiles == 1:
            return tensor.unsqueeze(0)

        return tensor.unsqueeze(0).repeat(num_tiles, 1, 1, 1)

        # raise NotImplementedError("Implement image preprocessing")

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """

        image_tokens = ' '.join([IMAGE_TOKEN] * self.config.num_image_tokens)

        visual_part = (
            f"{IMAGE_START_TOKEN} "
            f"{image_tokens} "
            f"{IMAGE_END_TOKEN}"
        )

        options = "\n".join(sample.options)

        prompt = (
            f"User:\n"
            f"{visual_part}\n\n"
            f"Question: {sample.question}\n\n"
            f"Options:\n{options}\n\n"
            f"Assistant:"
        )

        if include_answer:
            prompt += f" {sample.answer}"

        return prompt

        # raise NotImplementedError("Implement prompt construction")

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """

        prompt_text = self.build_prompt(sample, include_answer=False)
        answer_text = f'{sample.answer}'

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)

        input_ids = prompt_ids + answer_ids
        input_ids = input_ids[: self.config.max_length]

        attention_mask = [1] * len(input_ids)

        labels = [self.config.ignore_index] * len(prompt_ids) + answer_ids
        labels = labels[: len(input_ids)]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)}

        # raise NotImplementedError("Implement sample tokenization")

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """

        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        pixel_values = [b["pixel_values"] for b in batch]

        max_len = max(x.shape[0] for x in input_ids)

        def pad(seq, pad_value):
            return torch.nn.functional.pad(
                seq,
                (0, max_len - seq.shape[0]),
                value=pad_value,
            )

        input_ids = torch.stack([pad(x, self.tokenizer.pad_token_id) for x in input_ids])
        attention_mask = torch.stack([pad(x, 0) for x in attention_mask])
        labels = torch.stack([pad(x, self.config.ignore_index) for x in labels])

        pixel_values = torch.stack(pixel_values)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

        # raise NotImplementedError("Implement collate_fn")
