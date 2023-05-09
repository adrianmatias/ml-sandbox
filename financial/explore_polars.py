import pandas as pd
import polars as pl

df = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],
        "fruits": ["banana", "banana", "apple", "apple", "banana"],
        "B": [5, 4, 3, 2, 1],
        "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
    }
)

print(df)

# embarrassingly parallel execution & very expressive query language
df.sort("fruits").select(
    [
        "fruits",
        "cars",
        pl.lit("fruits").alias("literal_string_fruits"),
        pl.col("B").filter(pl.col("cars") == "beetle").sum(),
        pl.col("A").filter(pl.col("B") > 2).sum().over("cars").alias("sum_A_by_cars"),
        pl.col("A").sum().over("fruits").alias("sum_A_by_fruits"),
        pl.col("A").reverse().over("fruits").alias("rev_A_by_fruits"),
        pl.col("A").sort_by("B").over("fruits").alias("sort_A_by_B_by_fruits"),
    ]
)
