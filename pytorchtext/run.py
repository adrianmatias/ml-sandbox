import string

import numpy as np
import torch
from data import DatasetText, Tokenizer
from torch import nn
from torch.utils.data import DataLoader
from transformer_auto_regressive import TransformerAutoRegressive


class Conf:
    batch_size = 8
    seq_len = 4
    embedding_dim = 6
    attention_head_count = 3


def main():
    np.random.seed(0)
    text = "".join(
        np.random.choice(list(string.ascii_letters + " "), size=1000, replace=True)
    )
    tokenizer = Tokenizer()
    tokenizer.fit(text)

    dataset = DatasetText(
        token_id_list=tokenizer.tokenize(text),
        batch_size=Conf.batch_size,
        seq_len=Conf.seq_len,
    )

    model = TransformerAutoRegressive(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=Conf.embedding_dim,
        seq_len=Conf.seq_len,
        attention_head_count=Conf.attention_head_count,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(
        dataset=dataset, batch_size=Conf.batch_size, shuffle=False
    )

    iter = 0
    for epoch in range(3):
        for x, y in train_dataloader:
            print(x)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(x, y)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, y)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for x_eval, y_eval in dataset:
                    #######################
                    #  USE GPU FOR MODEL  #
                    #######################

                    # Forward pass only to get logits/output
                    outputs = model(x_eval)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += y_eval.size(0)

                    #######################
                    #  USE GPU FOR MODEL  #
                    #######################
                    # Total correct predictions
                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == y_eval.cpu()).sum()
                    else:
                        correct += (predicted == y_eval).sum()

                accuracy = 100 * correct / total

                # Print Loss
                print(
                    "Iteration: {}. Loss: {}. Accuracy: {}".format(
                        iter, loss.item(), accuracy
                    )
                )


if __name__ == "__main__":
    main()
