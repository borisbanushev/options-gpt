from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Dict, Any

from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter

from ..configs.config import FineTuningConfig
from ..tracking.weave_tracker import WeaveTracker
from ..data.schemas import TrainingExample


@dataclass
class MLXTrainer:
    cfg: FineTuningConfig
    tracker: WeaveTracker
    tensorboard_writer: SummaryWriter | None = None

    def _build_dataset(self, examples: Iterable[TrainingExample]) -> Dataset:
        records = [e.model_dump() for e in examples]
        return Dataset.from_list(records)

    def _formatting_func(self, example: Dict[str, Any]) -> str:
        return (
            f"Instruction: {example['instruction']}\n"
            f"Input: {example['input']}\n"
            f"Output: {example['output']}"
        )

    def train(self, train_examples: Iterable[TrainingExample], eval_examples: Iterable[TrainingExample] | None = None) -> None:
        # Imports inside to avoid hard dependency if mlx not installed yet
        from mlx_lm import load, utils
        import mlx.core as mx
        import mlx.optimizers as optim

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        # Initialize TensorBoard
        tensorboard_log_dir = os.path.join(self.cfg.train.output_dir, "tensorboard_logs_mlx")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        # Load model/tokenizer with mlx_lm
        model_id = self.cfg.train.model_id
        if os.getenv("MODEL_LOCAL_PATH"):
            model_id = os.getenv("MODEL_LOCAL_PATH")

        tokenizer, model = load(model_id)

        # Prepare data
        train_ds = self._build_dataset(train_examples)
        eval_ds = self._build_dataset(eval_examples) if eval_examples else None

        # Optimizer & scheduler (simple AdamW cosine)
        lr = self.cfg.train.learning_rate
        weight_decay = self.cfg.train.weight_decay
        optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

        # Collate and batching
        max_len = self.cfg.data.max_seq_length

        def encode_batch(texts: list[str]):
            toks = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors=None)
            input_ids = [mx.array(t) for t in toks["input_ids"]]
            attention_mask = [mx.array(t) for t in toks["attention_mask"]]
            return input_ids, attention_mask

        per_device_bs = self.cfg.train.per_device_train_batch_size
        grad_accum = self.cfg.train.gradient_accumulation_steps

        # Simple training loop
        global_step = 0
        model.train()
        num_epochs = self.cfg.train.num_train_epochs

        def dataset_iter(ds: Dataset):
            for ex in ds:
                yield self._formatting_func(ex)

        for epoch in range(int(num_epochs)):
            batch_texts: list[str] = []
            accum_loss = 0.0
            accum_count = 0
            for text in dataset_iter(train_ds):
                batch_texts.append(text)
                if len(batch_texts) < per_device_bs:
                    continue

                input_ids, attention_mask = encode_batch(batch_texts)
                batch_texts = []

                def loss_fn():
                    # mlx_lm models usually expose a generate/logits call via model
                    # Here we compute cross-entropy loss via utils.loss
                    return utils.loss(model, tokenizer, input_ids)

                loss, grads = mx.value_and_grad(loss_fn)()
                accum_loss += float(loss.item())
                accum_count += 1

                if accum_count % grad_accum == 0:
                    optimizer.update(model, grads)
                    optimizer.zero_grad(model)
                    global_step += 1

                    # Log
                    avg_loss = accum_loss / grad_accum
                    self.tensorboard_writer.add_scalar("train/loss", avg_loss, global_step)
                    self.tracker.log_metrics({"loss": avg_loss}, step=global_step)
                    accum_loss = 0.0

            # Simple eval
            if eval_ds:
                model.eval()
                eval_texts = [self._formatting_func(ex) for ex in eval_ds.select(range(min(64, len(eval_ds))))]
                if eval_texts:
                    input_ids, _ = encode_batch(eval_texts)
                    with mx.eval_mode():
                        eval_loss = float(utils.loss(model, tokenizer, input_ids).item())
                    self.tensorboard_writer.add_scalar("eval/loss", eval_loss, epoch)
                    self.tracker.log_metrics({"eval_loss": eval_loss}, step=global_step)
                model.train()

        # Save model
        save_dir = self.cfg.train.output_dir
        os.makedirs(save_dir, exist_ok=True)
        utils.save(model, tokenizer, save_dir)
        self.tracker.log_artifact(save_dir, name="fine_tuned_model_mlx", type_="model")

        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            print(f"✅ MLX training complete! TensorBoard logs saved to: {tensorboard_log_dir}")


