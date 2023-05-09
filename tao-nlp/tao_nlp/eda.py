from pprint import pprint

import pandas as pd
import seaborn as sns
from datasets import Dataset, load_dataset
from matplotlib import pyplot as plt
from pandas import DataFrame

sns.set_theme()


class Conf:
    col_label = "label"
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"


def main():
    ds = load_dataset("imdb")
    pprint(ds["train"][0])
    pprint(ds["test"][-100])

    df_train = DataFrame(ds["train"])
    df_test = DataFrame(ds["test"])

    print(df_train[Conf.col_label].value_counts())
    print(df_test[Conf.col_label].value_counts())

    describe_char_count(df=df_train)
    describe_char_count(df=df_test)
    plt.show()

    get_token_mean_char_count(ds=ds["train"])


def describe_char_count(df: DataFrame):
    df["char_count"] = df["text"].str.len()
    sns.displot(df["char_count"])


def get_token_mean_char_count(ds: Dataset):
    import numpy as np
    from transformers import DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained(Conf.model_name)

    text_sample = ds
    # Tokenize the text
    tokens = tokenizer.tokenize(". ".join(ds["text"][:100]))

    # Calculate the mean character length of the tokens
    mean_token_length = np.mean([len(token) for token in tokens])

    print(f"Mean character length of the tokens: {mean_token_length:.2f}")


if __name__ == "__main__":
    main()
