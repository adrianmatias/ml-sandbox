import shutil
import time
from typing import Tuple

import torch


class BigramLanguageModel(torch.nn.Module):
    def generate(self, contexts: torch.Tensor, max_new_tokens: int):
        # contexts is of size (B, T), it's the array of indices of the current context.
        for _ in range(max_new_tokens):
            logits, _ = self(
                contexts[:, -self.block_size :]
            )  # Get the predictions. (B, T, C)
            # We need to trim the time to the last at most block_size tokens due to the positional encoder.
            logits = logits[
                :, -1, :
            ]  # We focus on the last element only, as it's what we want.
            # It becomes size (B, C)
            probs = torch.softmax(
                logits, dim=-1
            )  # We transform the logits to probabilities.
            idx_next = torch.multinomial(
                probs, num_samples=1
            )  # We sample the next token, (B,1)
            contexts = torch.cat((contexts, idx_next), dim=1)  # (B, T+1)
        return contexts

    def generate_forever(self, decoder, sleep_time: float):
        message = ""
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        try:
            while True:
                context = self.generate(context[:, -self.block_size :], 1)
                next_symbol = decoder([context[0][-1].tolist()])
                if next_symbol == "\n":
                    message = ""
                    print("")
                else:
                    width = shutil.get_terminal_size()[0]
                    if len(message) == width:
                        message = next_symbol
                        print("")
                    else:
                        message += next_symbol
                    print(f"{message}", end="\r")
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            return

    def train_model(
        self,
        number_iterations: int,
        train_data: torch.Tensor,
        validation_data: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        eval_interval: int,
        eval_iters: int,
    ):
        last_val_loss = float("inf")
        iter = 0
        try:
            while True:
                # We evaluate loss on train and val once in a while
                if iter % eval_interval == 0 or iter == number_iterations - 1:
                    losses = estimate_loss(
                        self,
                        eval_iters,
                        self.block_size,
                        batch_size,
                        train_data,
                        validation_data,
                    )
                    print(
                        f"At step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}."
                    )
                    if losses["val"] >= last_val_loss:
                        print("Early stop.")
                        return
                    last_val_loss = losses["val"]

                # Get a batch of data
                contexts, targets = get_batch(
                    "train", self.block_size, batch_size, train_data, validation_data
                )

                # Optimize
                _, loss = self(contexts, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                iter += 1
                if iter == number_iterations:
                    return

        except (
            KeyboardInterrupt
        ):  # Allows for keyboard interruption without exiting the script.
            print("Training manually interrupted.")


# Data loading
def get_batch(
    split: str,
    block_size: int,
    batch_size: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else validation_data
    indices_start = torch.randint(len(data) - block_size - 1, (batch_size,))
    contexts = torch.stack([data[i : i + block_size] for i in indices_start])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in indices_start])
    return contexts, targets


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    eval_iters: int,
    block_size: int,
    batch_size: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
) -> dict:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            contexts, targets = get_batch(
                split, block_size, batch_size, train_data, validation_data
            )
            _, loss = model(contexts, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
