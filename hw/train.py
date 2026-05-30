from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hw.dataset import MathVQADataset
from hw.model import MathVLM
from hw.processor import MathVLMProcessor, ProcessorConfig

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import ViTModel

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
language_model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
vision_encoder = ViTModel.from_pretrained("hf-internal-testing/tiny-random-vit")

from hw.model import ModelConfig

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """

    model.train()
    optimizer.zero_grad()

    outputs = model(batch)

    # FIX: dict-safe access
    loss = outputs["loss"]

    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss: {loss.item()}")

    loss.backward()
    optimizer.step()

    return float(loss.item())

# raise NotImplementedError("Implement train_one_step")


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """

    model_config = ModelConfig(
        vision_hidden_size=vision_encoder.config.hidden_size,
        text_hidden_size=language_model.config.n_embd,
        num_image_tokens=config["processor"]["num_image_tokens"],
        image_token_id=tokenizer.convert_tokens_to_ids("<image>"),
    )

    device = torch.device(config["trainer"]["device"])

    dataset = MathVQADataset(
        manifest_path=config["data"]["train_manifest"],
        split=config["data"]["split"],
        max_samples=config["data"]["max_samples"],
    )

    processor = MathVLMProcessor(
        tokenizer=tokenizer,
        config=ProcessorConfig(
            image_size=config["processor"]["image_size"],
            num_tiles=config["processor"]["num_tiles"],
            tile_overlap=config["processor"]["tile_overlap"],
            num_image_tokens=config["processor"]["num_image_tokens"],
            max_length=config["processor"]["max_length"],
            ignore_index=config["processor"]["ignore_index"],
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=config["trainer"]["local_batch_size"],
        shuffle=True,
        num_workers=config["trainer"]["num_workers"],
        collate_fn=lambda batch: processor.collate([processor(sample) for sample in batch]),
    )

    model = MathVLM(
        vision_encoder=vision_encoder,
        language_model=language_model,
        config=model_config,
    )

    if config["model"]["freeze_vision"]:
        for p in model.vision_encoder.parameters():
            p.requires_grad = False

    if config["model"]["freeze_llm"]:
        for p in model.language_model.parameters():
            p.requires_grad = False

    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["trainer"]["learning_rate"],
        weight_decay=config["trainer"]["weight_decay"],
    )

    local_bs = config["trainer"]["local_batch_size"]
    global_bs = config["trainer"]["global_batch_size"]

    grad_acc_steps = max(1, global_bs // local_bs)

    max_steps = config["trainer"]["max_steps"]

    if fast_train:
        max_steps = min(max_steps, 2)

    global_step = 0

    optimizer.zero_grad()

    for epoch in range(config["trainer"]["num_train_epochs"]):

        for batch in loader:

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch)

            loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss became {loss.item()}")

            (loss / grad_acc_steps).backward()

            if (global_step + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            print(
                f"step={global_step} "
                f"loss={loss.item():.4f}"
            )

            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    save_path = config["trainer"]["save_checkpoint_path"]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True,)

        torch.save({"adapter": model.adapter.state_dict(), "config": config}, save_path)

        print(f"Checkpoint saved to {save_path}")

    # raise NotImplementedError("Implement run_training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
