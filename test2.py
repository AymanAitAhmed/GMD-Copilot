import pandas as pd

examples = pd.read_csv(r'.\data\few_shot_examples.csv').to_dict(orient="records")
print(examples)
