import pandas as pd

examples = pd.read_csv(r'.\data\few_shot_examples.csv').to_dict(orient="records")

documentation = [
    "our business defines a cycle as going from a state = 35 to the next state = 35, so like this 35,36,37,...,34,35",
    "our business considers a part has been manufactured in table 10 if a state=35 but in table 07 and table 03 its if state=30 has been registered in table_states",
    "our business considers production in table 10 to be perfect if its 22 parts a day, acceptable if its 20 or 21 parts a day and not acceptable for anything below 20",
    "our business only works from 8AM to 17PM.",
]


api_key = "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"

logo_path = "https://i.ibb.co/FL3mpvWR/gmd-logo.png"

tmp_file_dir = "webApp/tmp/"

app_secret_key = b'_5#y2L"F4Q8z\n\xec]/'

chromadb_path = r"webApp\chromadb"


