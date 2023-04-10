import torch
from bigram_language_model_karpathy import BigramLanguageModelKarpathy
from bigram_language_model_torch_layers import BigramLanguageModelTorchLayers

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    is_model_karpathy = False

    batch_size = 32
    block_size = 128
    max_iters = 2
    eval_interval = 10
    learning_rate = 3e-4
    dropout = 0.2
    eval_iters = 200
    number_layers = 2
    number_heads = 4
    number_embeddings = number_heads * 32  # 384 / 6 = 64 dimensional heads

    torch.manual_seed(1337)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open("input.txt", "r", encoding="utf-8") as f:
        corpus = f.read()

    # The properties of the test
    chars = sorted(list(set(corpus)))
    vocab_size = len(chars)

    # Encoding and decoding
    string_to_int = {character: index for index, character in enumerate(chars)}
    int_to_string = {index: character for index, character in enumerate(chars)}
    encode = lambda string: [string_to_int[char] for char in string]
    decode = lambda list_int: "".join([int_to_string[integer] for integer in list_int])

    # Train and test splits
    data = torch.tensor(encode(corpus), dtype=torch.long, device=device)
    number_train = int(0.9 * len(data))
    train_data = data[:number_train]
    validation_data = data[:number_train]

    if is_model_karpathy:
        model = BigramLanguageModelKarpathy(
            vocab_size=vocab_size,
            number_embeddings=number_embeddings,
            block_size=block_size,
            number_heads=number_heads,
            number_layers=number_layers,
            dropout=dropout,
            device=device,
        ).to(device)
        """
        At step 0: train loss 4.2914, val loss 4.2902.
        At step 10: train loss 3.6093, val loss 3.6106.
        At step 20: train loss 3.3169, val loss 3.3162.
        At step 30: train loss 3.1954, val loss 3.1960.
        At step 40: train loss 3.1091, val loss 3.1072.
        At step 50: train loss 3.0234, val loss 3.0236.
        At step 60: train loss 2.9519, val loss 2.9542.
        At step 70: train loss 2.8861, val loss 2.8906.
        At step 80: train loss 2.8380, val loss 2.8407.
        At step 90: train loss 2.7977, val loss 2.7973.
        At step 99: train loss 2.7672, val loss 2.7666.
        """
    else:
        model = BigramLanguageModelTorchLayers(
            vocab_size=vocab_size,
            number_embeddings=number_embeddings,
            block_size=block_size,
            number_heads=number_heads,
            number_layers=number_layers,
            dropout=dropout,
            device=device,
        ).to(device)
        """
        At step 0: train loss 4.4070, val loss 4.4057.
        At step 10: train loss 3.4882, val loss 3.4909.
        At step 20: train loss 3.3020, val loss 3.3015.
        At step 30: train loss 3.1780, val loss 3.1793.
        At step 40: train loss 3.0504, val loss 3.0485.
        At step 50: train loss 2.9327, val loss 2.9345.
        At step 60: train loss 2.8489, val loss 2.8509.
        At step 70: train loss 2.7756, val loss 2.7812.
        At step 80: train loss 2.7257, val loss 2.7296.
        At step 90: train loss 2.6821, val loss 2.6830.
        At step 99: train loss 2.6528, val loss 2.6529.
        """

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Train
    model.train_model(
        max_iters,
        train_data,
        validation_data,
        optimizer,
        batch_size,
        eval_interval,
        eval_iters,
    )

    # Generate from the model
    print("\nAn example of text generated from the model.")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, 300)[0].tolist()))
    print("\nNow we generate text forever.\n")
    model.generate_forever(decode, 0.1)


# Data loading
def get_batch(
    split: str,
    block_size: int,
    batch_size: int,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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


if __name__ == "__main__":
    main()
