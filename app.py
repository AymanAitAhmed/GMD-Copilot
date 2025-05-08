from typing import List
import os

from openai import OpenAI

from webApp.llm_integrations.llm import OpenAI_Chat
from webApp.my_chromadb.chromadb import ChromaDB_VectorStore
from FlaskApp import FlaskApp
from constants import *


open_router = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


class BaseApp(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=open_router, config=config)
        self.collections = {
            "documentation": self.documentation_collection,
            "ddl": self.ddl_collection,
            "sql": self.sql_collection,
        }


base_app = BaseApp(config={
    'model': 'microsoft/mai-ds-r1:free',
    "path": chromadb_path,
    "client": "persistent",
    "temperature": 0,
    "language": "French",
    "n_results_sql": 6
})

try:
    base_app.connect_to_database(host='192.168.7.236', dbname='iot2050db', user='iot2050', password='iot2050iot',
                                 port='5432')
except:
    db_host = '127.0.0.1'
    db_port = 5432
    db_user = 'gmd'
    db_password = '1234'
    db_name = 'gmd_iot'
    base_app.connect_to_database(host=db_host, dbname=db_name, user=db_user, password=db_password,
                                 port='5432')


def init_vector_db(base_app, include_examples=True):
    if not include_examples:
        train_df = base_app.get_training_data()
        documentation_ids = train_df[train_df['training_data_type'] == 'documentation'].id.tolist()
        for id in documentation_ids:
            base_app.remove_training_data(id=id)

    df_information_schema = base_app.run_sql("SELECT table_name,column_name, data_type FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema='public'")

    plan = base_app.get_training_plan_generic(df_information_schema)
    base_app.train(plan=plan)

    if include_examples:
        for example in examples:
            base_app.train(sql=example["content"], question=example["question"])

    for doc in documentation:
        base_app.train(documentation=doc)

    train_df = base_app.get_training_data()
    print("init was successful, ", train_df.shape[0], "docs exist in the database")


create_first_vector_db = len(os.listdir(chromadb_path)) <= 1

# if create_first_vector_db:
#     init_vector_db(base_app=base_app)
# else:
#     init_vector_db(base_app=base_app, include_examples=False)

app = FlaskApp(base_app, allow_llm_to_see_data=True, debug=True,
               logo=logo_path,
               title='GMD Copilot', summarization=True, ask_results_correct=True,
               subtitle='Your AI-powered copilot for extracting insights from Data.',
               show_training_data=False, sql=False, max_attempts=5,
               app_secret_key=app_secret_key,
               # index_html_path="index.html"
               )
if __name__ == "__main__":
    app.run()
