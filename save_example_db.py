import json
import pandas as pd
import requests


ls = json.loads(requests.get("http://localhost:8084/api/v0/get_training_data").json()['df'])
ids = [element['id'] for element in ls]

df = pd.DataFrame(columns=['question', 'content'])
for element in ls:
    if element['training_data_type'] == 'sql':
        df.loc[-1] = [element['question'], element['content']]
        df.index = df.index + 1
        df = df.sort_index()

df.to_csv(r'C:\Users\ACER\PycharmProjects\LogisticRegression\env\text2sql\data\few_shot_examples.csv', index=False)
