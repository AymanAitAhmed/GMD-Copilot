import json
import logging
import traceback
import os
import sys
import uuid
import hashlib
import time
from abc import ABC, abstractmethod
from functools import wraps
import importlib.metadata
import markdown
import re

import pandas as pd
import duckdb
import flask
from flask_cors import CORS
import requests
from flasgger import Swagger
from flask import Flask, Response, jsonify, request, send_from_directory, session, make_response, flash
from werkzeug.utils import secure_filename
from flask_sock import Sock

from webApp.base.base import CopilotBase
from webApp.assets import *
from webApp.auth import AuthInterface, NoAuth
from constants import tmp_file_dir


class Cache(ABC):
    """
    Define the interface for a cache that can be used to store data in a Flask app.
    """

    @abstractmethod
    def generate_id(self, *args, **kwargs):
        """
        Generate a unique ID for the cache.
        """
        pass

    @abstractmethod
    def get(self, id, field):
        """
        Get a value from the cache.
        """
        pass

    @abstractmethod
    def get_all(self, field_list) -> list:
        """
        Get all values from the cache.
        """
        pass

    @abstractmethod
    def set(self, id, field, value):
        """
        Set a value in the cache.
        """
        pass

    @abstractmethod
    def delete(self, id):
        """
        Delete a value from the cache.
        """
        pass


class MemoryCache(Cache):
    def __init__(self):
        self.cache = {}

    def generate_id(self, *args, **kwargs):
        return str(uuid.uuid4())

    def set(self, id, field, value):
        if id not in self.cache:
            self.cache[id] = {}

        self.cache[id][field] = value

    def get(self, id, field):
        if id not in self.cache:
            return None

        if field not in self.cache[id]:
            return None

        return self.cache[id][field]

    def get_all(self, field_list) -> list:
        return [
            {"id": id, "name": "this is a mock title", "created_at": time.time(),
             **{field: self.get(id=id, field=field) for field in field_list}}
            for id in self.cache
        ]

    def delete(self, id):
        if id in self.cache:
            del self.cache[id]


class FlaskAPI:
    flask_app = None

    def requires_cache(self, required_fields, optional_fields=[]):
        def decorator(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                id = request.args.get("id")

                if id is None:
                    id = request.json.get("id")
                    if id is None:
                        return jsonify({"type": "error", "error": "No id provided"})

                for field in required_fields:
                    if self.cache.get(id=id, field=field) is None:
                        return jsonify({"type": "error", "error": f"No {field} found"})

                field_values = {
                    field: self.cache.get(id=id, field=field) for field in required_fields
                }

                for field in optional_fields:
                    field_values[field] = self.cache.get(id=id, field=field)

                # Add the id to the field_values
                field_values["id"] = id

                return f(*args, **field_values, **kwargs)

            return decorated

        return decorator

    def requires_auth(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user = self.auth.get_user(flask.request)

            if not self.auth.is_logged_in(user):
                return jsonify({"type": "not_logged_in", "html": self.auth.login_form()})

            # Pass the user to the function
            return f(*args, user=user, **kwargs)

        return decorated

    def __init__(
            self,
            copilot: CopilotBase,
            cache: Cache = MemoryCache(),
            auth: AuthInterface = NoAuth(),
            debug=True,
            allow_llm_to_see_data=False,
            chart=True,
            max_generated_question_length=200,
            max_attempts=3,
            app_secret_key=None
    ):
        """
        Expose a Flask API that can be used to interact with a Copilot instance.

        Args:
            copilot: The Copilot instance to interact with.
            cache: The cache to use. Defaults to MemoryCache, which uses an in-memory cache. You can also pass in a custom cache that implements the Cache interface.
            auth: The authentication method to use. Defaults to NoAuth, which doesn't require authentication. You can also pass in a custom authentication method that implements the AuthInterface interface.
            debug: Show the debug console. Defaults to True.
            allow_llm_to_see_data: Whether to allow the LLM to see data. Defaults to False.
            chart: Whether to show the chart output in the UI. Defaults to True.

        Returns:
            None
        """

        self.flask_app = Flask(__name__)
        CORS(
            self.flask_app,
            supports_credentials=True,
            resources={
                r"/api/*": {
                    "origins": "http://localhost:3001"
                }
            }
        )

        self.swagger = Swagger(
            self.flask_app, template={"info": {"title": "Copilot API"}}
        )
        self.sock = Sock(self.flask_app)
        self.ws_clients = []
        self.copilot = copilot
        self.auth = auth
        self.cache = cache
        self.debug = debug
        self.allow_llm_to_see_data = allow_llm_to_see_data
        self.chart = chart
        self.config = {
            "debug": debug,
            "allow_llm_to_see_data": allow_llm_to_see_data,
            "chart": chart,
        }
        self.max_generated_question_length = max_generated_question_length
        self.max_attempts = max_attempts
        self.app_secret_key = app_secret_key
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.INFO)

        if self.debug:
            def log(message, title="Info"):
                [ws.send(json.dumps({'message': message, 'title': title})) for ws in self.ws_clients]

            self.copilot.log = log

        @self.flask_app.route("/api/v0/get_config", methods=["GET"])
        @self.requires_auth
        def get_config(user: any):
            """
            Get the configuration for a user
            ---
            parameters:
              - name: user
                in: query
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: config
                    config:
                      type: object
            """
            config = self.auth.override_config_for_user(user, self.config)
            return jsonify(
                {
                    "type": "config",
                    "config": config
                }
            )

        @self.flask_app.route("/api/v0/generate_questions", methods=["GET"])
        @self.requires_auth
        def generate_questions(user: any):
            """
            Generate questions
            ---
            parameters:
              - name: user
                in: query
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: question_list
                    questions:
                      type: array
                      items:
                        type: string
                    header:
                      type: string
                      default: Here are some questions you can ask
            """

            training_data = copilot.get_training_data()

            # If training data is None or empty
            if training_data is None or len(training_data) == 0:
                return jsonify(
                    {
                        "type": "error",
                        "error": "No training data found. Please add some training data first.",
                    }
                )

            # Get the questions from the training data
            try:
                # Filter training data to only include questions where the question is not null
                # "and smaller than a max length(not included in the original implementation added by me)"
                filtered_training_data = training_data[
                    training_data["question"].str.len() < self.max_generated_question_length]
                questions = (
                    filtered_training_data[filtered_training_data["question"].notnull()]
                    .sample(3)["question"]
                    .tolist()
                )

                # Temporarily this will just return an empty list
                return jsonify(
                    {
                        "type": "question_list",
                        "questions": questions,
                        "header": "Here are some questions you can ask",
                    }
                )
            except Exception as e:
                return jsonify(
                    {
                        "type": "question_list",
                        "questions": [],
                        "header": "Go ahead and ask a question",
                    }
                )

        @self.flask_app.route("/api/v0/upload_spreadsheet_file", methods=["POST"])
        @self.requires_auth
        def upload_spreadsheet_file(user: any):

            # Find all file keys that start with "file"
            file_keys = [key for key in request.files if key.startswith('file')]
            if not file_keys:
                return jsonify({"type": "error", "error": "No file provided"})

            ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

            def allowed_file(filename):
                return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

            session_id = request.cookies.get('session_id', 'unknown')
            if not session_id or session_id == 'unknown':  # Ensure session_id is valid
                return jsonify({"type": "error", "error": "Invalid session."})

            if not self.copilot.con:
                return jsonify({"type": "error", "error": "Database connection not available."})

            results = []
            for key in file_keys:
                file = request.files.get(key)
                if not file or file.filename == '':
                    return jsonify({"type": "error", "error": "No file provided"})

                if not allowed_file(file.filename):
                    return jsonify({"type": "error",
                                    "error": f"File '{file.filename}': only files with extensions {ALLOWED_EXTENSIONS} are allowed"})

                # Build a secure filename using the session_id cookie
                filename = secure_filename("file_" + session_id + "_" + file.filename)
                file_path = os.path.join(tmp_file_dir, filename)

                try:
                    file.save(file_path)
                    results.append(f"File '{file.filename}' uploaded successfully")
                    # --adding the file to duckdb--
                    df = None
                    if file.filename.lower().endswith('.csv'):
                        df = pd.read_csv(file.stream)
                    else:
                        excel_file = pd.ExcelFile(file.stream)
                        if excel_file.sheet_names:
                            sheet_name = excel_file.sheet_names[0]
                            df = pd.read_excel(excel_file, sheet_name=sheet_name)
                        else:
                            results.append(f"File '{file.filename}': No sheets found in Excel file.")
                            continue

                    if df is not None and not df.empty:

                        current_session_tables = self.copilot.con.sql(
                            f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'temp'").fetchall()
                        table_index = len(current_session_tables)
                        unique_table_name = f"table_{table_index}"

                        self.copilot.con.sql(
                            f"""CREATE OR REPLACE TEMP TABLE "{unique_table_name}" AS SELECT * FROM df;""")
                        results.append(
                            f"File '{file.filename}' loaded as temporary table '{unique_table_name}'.")

                    else:
                        results.append(f"File '{file.filename}' was empty or could not be read into a DataFrame.")

                except Exception as e:
                    results.append(f"Error saving file '{filename}': {str(e)}")

            return jsonify({"type": "text", "text": results})

        def get_uploaded_spreadsheets():
            """
            Reads only CSV and Excel files from a directory whose filenames contain
            the session_id stored in Flask cookies.

            Parameters:
                dir_path (str): Path to the directory containing the files.
                session_id (str): The session ID to filter files.

            Returns:
                list: A list containing pandas DataFrames.
            """
            dataframes = []

            if not os.path.isdir(tmp_file_dir):
                raise ValueError(f"The provided path '{tmp_file_dir}' is not a valid directory.")

            session_value = flask.request.cookies.get("session_id")
            if not session_value:
                print("Session ID not found in cookies.")
                return dataframes

            # List files in the directory and filter based on session ID
            files = [f for f in os.listdir(tmp_file_dir) if session_value in f]
            if not files:
                return dataframes

            for file in files:
                file_path = os.path.join(tmp_file_dir, file)
                if not os.path.isfile(file_path):
                    continue

                if file.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        dataframes.append(df)
                    except Exception as e:
                        print(f"Error reading CSV file '{file}': {e}")
                elif file.lower().endswith(('.xlsx', '.xls')):
                    try:
                        excel_file = pd.ExcelFile(file_path)
                        for sheet_name in excel_file.sheet_names:
                            try:
                                df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
                                dataframes.append(df_sheet)
                            except Exception as e:
                                print(f"Error reading sheet '{sheet_name}' in Excel file '{file}': {e}")
                    except Exception as e:
                        print(f"Error processing Excel file '{file}': {e}")
                else:
                    print(f"Skipping unsupported file type: {file}")

            return dataframes

        @self.flask_app.route("/api/v0/delete_uploaded_file", methods=["POST"])
        @self.requires_auth
        def delete_uploaded_file(user: any):
            # Extract JSON data from the request
            data = request.get_json()
            if not data or 'fileName' not in data:
                return jsonify({"type": "error", "error": "No fileName provided"}), 400

            original_file_name = data['fileName']

            # Retrieve session_id cookie to rebuild the secure filename
            session_id = request.cookies.get('session_id', 'unknown')
            secure_name = secure_filename("file_" + session_id + "_" + original_file_name)
            file_path = os.path.join(tmp_file_dir, secure_name)

            # Check if the file exists
            if not os.path.exists(file_path):
                return jsonify({"type": "error", "error": f"File '{secure_name}' not found"}), 404

            try:
                os.remove(file_path)
                return jsonify({"type": "text", "text": f"File '{original_file_name}' deleted successfully"}), 200
            except Exception as e:
                return jsonify({"type": "error", "error": f"Error deleting file '{secure_name}': {str(e)}"}), 500

        @self.flask_app.route("/api/v0/delete_uploaded_files", methods=["POST"])
        @self.requires_auth
        def delete_uploaded_files(user: any):
            """
               Deletes only files in the temporary directory that contain the session ID
               stored in Flask cookies.

               Returns:
                   Flask Response: A JSON response indicating success or failure.
            """

            session_value = request.cookies.get("session_id")
            if not session_value:
                return jsonify({"type": "error", "text": "Session ID not found in cookies."}), 400

            for file in os.listdir(tmp_file_dir):
                if session_value in file:
                    file_path = os.path.join(tmp_file_dir, file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting file '{file}': {e}")
            return jsonify({"type": "text", "text": "deleted successfully"}), 200

        @self.flask_app.route("/api/v0/generate_sql", methods=["GET", "POST"])
        @self.requires_auth
        def generate_sql(user: any):
            """
            Generate SQL from a question
            ---
            parameters:
              - name: user
                in: query
              - name: question
                in: query
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: sql
                    id:
                      type: string
                    text:
                      type: string
            """

            question = flask.request.args.get("question")
            user_id = flask.request.cookies.get("session_id")

            if question is None or question == "":
                return jsonify({"type": "error", "error": "No question provided"})

            dfs = get_uploaded_spreadsheets()

            conversation_id = self.cache.generate_id(question=question)

            sql = copilot.generate_sql(question=question, allow_llm_to_see_data=self.allow_llm_to_see_data, dfs=dfs,
                                       user_id=user_id, conversation_id=conversation_id)

            self.cache.set(id=conversation_id, field="question", value=question)
            self.cache.set(id=conversation_id, field="sql", value=sql)

            if copilot.is_sql_valid(sql=sql):
                return jsonify(
                    {
                        "type": "sql",
                        "id": conversation_id,
                        "text": sql,
                    }
                )
            else:
                return jsonify(
                    {
                        "type": "text",
                        "id": conversation_id,
                        "text": sql,
                    }
                )

        @self.flask_app.route("/api/v0/generate_rewritten_question", methods=["GET"])
        @self.requires_auth
        def generate_rewritten_question(user: any):
            """
            Generate a rewritten question
            ---
            parameters:
              - name: last_question
                in: query
                type: string
                required: true
              - name: new_question
                in: query
                type: string
                required: true
            """

            last_question = flask.request.args.get("last_question")
            new_question = flask.request.args.get("new_question")

            rewritten_question = self.copilot.generate_rewritten_question(last_question, new_question)

            return jsonify({"type": "rewritten_question", "question": rewritten_question})

        @self.flask_app.route("/api/v0/get_function", methods=["GET"])
        @self.requires_auth
        def get_function(user: any):
            """
            Get a function from a question
            ---
            parameters:
              - name: user
                in: query
              - name: question
                in: query
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: function
                    id:
                      type: object
                    function:
                      type: string
            """
            question = flask.request.args.get("question")

            if question is None:
                return jsonify({"type": "error", "error": "No question provided"})

            if not hasattr(copilot, "get_function"):
                return jsonify({"type": "error", "error": "This setup does not support function generation."})

            id = self.cache.generate_id(question=question)
            function = copilot.get_function(question=question)

            if function is None:
                return jsonify({"type": "error", "error": "No function found"})

            if 'instantiated_sql' not in function:
                self.copilot.log(f"No instantiated SQL found for {question} in {function}")
                return jsonify({"type": "error", "error": "No instantiated SQL found"})

            self.cache.set(id=id, field="question", value=question)
            self.cache.set(id=id, field="sql", value=function['instantiated_sql'])

            if 'instantiated_post_processing_code' in function and function[
                'instantiated_post_processing_code'] is not None and len(
                function['instantiated_post_processing_code']) > 0:
                self.cache.set(id=id, field="plotly_code", value=function['instantiated_post_processing_code'])

            return jsonify(
                {
                    "type": "function",
                    "id": id,
                    "function": function,
                }
            )

        @self.flask_app.route("/api/v0/get_all_functions", methods=["GET"])
        @self.requires_auth
        def get_all_functions(user: any):
            """
            Get all the functions
            ---
            parameters:
              - name: user
                in: query
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: functions
                    functions:
                      type: array
            """
            if not hasattr(copilot, "get_all_functions"):
                return jsonify({"type": "error", "error": "This setup does not support function generation."})

            functions = copilot.get_all_functions()

            return jsonify(
                {
                    "type": "functions",
                    "functions": functions,
                }
            )

        @self.flask_app.route("/api/v0/run_sql", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["sql"])
        def run_sql(user: any, id: str, sql: str):
            """
            Run SQL
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: df
                    id:
                      type: string
                    df:
                      type: object
                    should_generate_chart:
                      type: boolean
            """

            attempt = 0
            last_error = None

            if not self.copilot.run_sql_is_set:
                return jsonify({
                    "type": "error",
                    "error": "Please connect to a database using copilot.connect_to_... in order to run SQL queries."
                })

            # uncomment if the uploading logic dont work
            # dfs = get_uploaded_spreadsheets()
            # if len(dfs) >= 1:
            #     for i, df in enumerate(dfs):
            #         self.copilot.con.sql(f"""CREATE OR REPLACE TEMP TABLE table{i} AS SELECT * FROM df;""")

            while attempt < self.max_attempts:
                try:
                    df_returned = self.copilot.con.sql(query=sql).df()

                    self.cache.set(id=id, field="df", value=df_returned)
                    self.cache.set(id=id, field="sql", value=sql)

                    return jsonify({
                        "type": "df",
                        "id": id,
                        "df": df_returned.head(10).to_json(orient='records', date_format='iso'),
                        "should_generate_chart": self.chart and copilot.should_generate_chart(df_returned),
                    })
                except Exception as e:
                    last_error = str(e)
                    attempt += 1

                    if attempt < self.max_attempts:
                        original_question = self.cache.get(id, "question")
                        fix_question = f"I have an error: {last_error}\n\nHere is the SQL I tried to run: {sql}\n\nThis is the question I was trying to answer: {original_question}\n\nCan you rewrite the SQL to fix the error?"

                        print(f"error in Generated SQL, Fixing the SQL, Attempt:{attempt}/{self.max_attempts}")
                        sql = copilot.generate_sql(question=fix_question, dfs=dfs)

                        self.cache.set(id=id, field="sql", value=sql)
                    else:
                        print(traceback.format_exc())
                        return jsonify({"type": "sql_error",
                                        "error": f"The Copilot tried fixing the error for {max_attempts} times but didn't succeed."})

        @self.flask_app.route("/api/v0/fix_sql", methods=["POST"])
        @self.requires_auth
        @self.requires_cache(["question", "sql"])
        def fix_sql(user: any, id: str, question: str, sql: str):
            """
            Fix SQL
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
              - name: error
                in: body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: sql
                    id:
                      type: string
                    text:
                      type: string
            """
            error = flask.request.json.get("error")

            if error is None:
                return jsonify({"type": "error", "error": "No error provided"})

            question = f"I have an error: {error}\n\nHere is the SQL I tried to run: {sql}\n\nThis is the question I was trying to answer: {question}\n\nCan you rewrite the SQL to fix the error?"

            dfs = get_uploaded_spreadsheets()

            fixed_sql = copilot.generate_sql(question=question, dfs=dfs)

            self.cache.set(id=id, field="sql", value=fixed_sql)

            return jsonify(
                {
                    "type": "sql",
                    "id": id,
                    "text": fixed_sql,
                }
            )

        @self.flask_app.route('/api/v0/update_sql', methods=['POST'])
        @self.requires_auth
        @self.requires_cache([])
        def update_sql(user: any, id: str):
            """
            Update SQL
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
              - name: sql
                in: body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: sql
                    id:
                      type: string
                    text:
                      type: string
            """
            sql = flask.request.json.get('sql')

            if sql is None:
                return jsonify({"type": "error", "error": "No sql provided"})

            self.cache.set(id=id, field='sql', value=sql)

            return jsonify(
                {
                    "type": "sql",
                    "id": id,
                    "text": sql,
                })

        @self.flask_app.route("/api/v0/download_csv", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df"])
        def download_csv(user: any, id: str, df):
            """
            Download CSV
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
            responses:
              200:
                description: download CSV
            """
            csv = df.to_csv()

            return Response(
                csv,
                mimetype="text/csv",
                headers={"Content-disposition": f"attachment; filename={id}.csv"},
            )

        @self.flask_app.route("/api/v0/generate_plotly_figure", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df", "question", "sql"])
        def generate_plotly_figure(user: any, id: str, df: pd.DataFrame, question, sql):
            """
            Generate plotly figure
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
              - name: chart_instructions
                in: body
                type: string
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: plotly_figure
                    id:
                      type: string
                    fig:
                      type: object
            """
            chart_instructions = flask.request.args.get('chart_instructions')
            try:
                code = copilot.generate_common_plotly(df)
                if code == "" and (chart_instructions is None or len(chart_instructions) == 0):
                    code = copilot.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                        df_sample=df.head(3)
                    )
                    self.cache.set(id=id, field="plotly_code", value=code)
                elif code == "":
                    question = f"{question}. When generating the chart, use these special instructions: {chart_instructions}"
                    code = copilot.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                        df_sample=df.head(3)
                    )
                    self.cache.set(id=id, field="plotly_code", value=code)

                fig = copilot.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
                fig_json = fig.to_json()

                self.cache.set(id=id, field="fig_json", value=fig_json)

                return jsonify(
                    {
                        "type": "plotly_figure",
                        "id": id,
                        "fig": fig_json,
                    }
                )
            except Exception as e:
                # Print the stack trace
                import traceback

                traceback.print_exc()

                return jsonify({"type": "error", "error": str(e)})

        @self.flask_app.route("/api/v0/get_training_data", methods=["GET"])
        @self.requires_auth
        def get_training_data(user: any):
            """
            Get all training data
            ---
            parameters:
              - name: user
                in: query
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: df
                    id:
                      type: string
                      default: training_data
                    df:
                      type: object
            """
            df = copilot.get_training_data()
            copilot.get_tables_to_use()
            if df is None or len(df) == 0:
                return jsonify(
                    {
                        "type": "error",
                        "error": "No training data found. Please add some training data first.",
                    }
                )

            return jsonify(
                {
                    "type": "df",
                    "id": "training_data",
                    "df": df.to_json(orient="records"),
                }
            )

        @self.flask_app.route("/api/v0/remove_training_data", methods=["POST"])
        @self.requires_auth
        def remove_training_data(user: any):
            """
            Remove training data
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
            """
            # Get id from the JSON body
            id = flask.request.json.get("id")

            if id is None:
                return jsonify({"type": "error", "error": "No id provided"})

            if copilot.remove_training_data(id=id):
                return jsonify({"success": True})
            else:
                return jsonify(
                    {"type": "error", "error": "Couldn't remove training data"}
                )

        @self.flask_app.route("/api/v0/run_agent", methods=["POST"])
        def run_agent_endpoint() -> any:
            data = request.get_json()
            question = data.get("question")
            if not question:
                return jsonify({"error": "Missing 'question' in request body."}), 400

            user_id = flask.request.cookies.get("session_id")
            thread_id = data.get("thread_id") or None  # Ensure it's None if empty
            print("cookies: ",flask.request.cookies.to_dict())
            print("thread_id from request: ", thread_id)
            print("user_id: ", user_id)
            
            # Use the new method to get or create a conversation ID
            # This preserves None values which trigger _build_initial_messages later
            conversation_id = self.copilot.get_or_create_conversation_id(thread_id, user_id, question)
            
            # Store task info in cache for status endpoint and streaming
            self.cache.set(conversation_id, "status", "UNDERSTANDING")
            self.cache.set(conversation_id, "type", "TEXT_TO_SQL")
            self.cache.set(conversation_id, "question", question)
            self.cache.set(conversation_id, "user_id", user_id)
            self.cache.set(conversation_id, "thread_id", thread_id)  # Store original thread_id if any
            
            # Return initial response with queryId to start the process
            return jsonify({
                "askingTask": {
                    "queryId": conversation_id,
                    "status": "UNDERSTANDING",
                    "type": "TEXT_TO_SQL",
                    "question": question,
                    "retrievedTables": [],
                    "sqlGenerationReasoning": ""
                }
            })
            
        @self.flask_app.route("/api/v0/api/ask_task/streaming", methods=["GET"])
        def streaming_endpoint():
            query_id = request.args.get('queryId')
            if not query_id:
                return jsonify({"error": "Missing queryId parameter"}), 400
                
            # Get stored info from cache
            question = self.cache.get(query_id, "question")
            user_id = self.cache.get(query_id, "user_id")
            thread_id = self.cache.get(query_id, "thread_id")  # Get original thread_id if any
            
            if not question or not user_id:
                return jsonify({"error": "Invalid queryId or missing data"}), 404
                
            # Update status to SEARCHING
            self.cache.set(query_id, "status", "SEARCHING")
            
            # Use the streaming method from LLM integration
            # Important: We pass thread_id here, which might be None
            # This preserves the correct conversation creation logic
            def generate():
                # Pass iterator results from stream_response to the client
                for response in self.copilot.stream_response(question, user_id, thread_id):
                    yield f"data: {response}\n\n"
                    
                    # Parse response to extract message and done status
                    try:
                        response_data = json.loads(response)
                        if response_data.get('done', False):
                            # Update status to FINISHED when done
                            self.cache.set(query_id, "status", "FINISHED")
                    except json.JSONDecodeError:
                        pass
                    
            return Response(generate(), mimetype='text/event-stream')
            
        @self.flask_app.route("/api/v0/api/ask_task/status", methods=["GET"])
        def task_status_endpoint():
            query_id = request.args.get('queryId')
            if not query_id:
                return jsonify({"error": "Missing queryId parameter"}), 400
                
            # Retrieve task info from cache
            status = self.cache.get(query_id, "status") or "UNDERSTANDING"
            task_type = self.cache.get(query_id, "type") or "TEXT_TO_SQL"
            question = self.cache.get(query_id, "question") or ""
            
            # Build response with available data
            response = {
                "askingTask": {
                    "queryId": query_id,
                    "status": status,
                    "type": task_type,
                    "question": question,
                    "retrievedTables": [],
                    "sqlGenerationReasoning": ""
                }
            }
            
            return jsonify(response)

        @self.flask_app.route("/api/v0/train", methods=["POST"])
        @self.requires_auth
        def add_training_data(user: any):
            """
            Add training data
            ---
            parameters:
              - name: user
                in: query
              - name: question
                in: body
                type: string
              - name: sql
                in: body
                type: string
              - name: ddl
                in: body
                type: string
              - name: documentation
                in: body
                type: string
            responses:
              200:
                schema:
                  type: object
                  properties:
                    id:
                      type: string
            """
            question = flask.request.json.get("question")
            sql = flask.request.json.get("sql")
            ddl = flask.request.json.get("ddl")
            documentation = flask.request.json.get("documentation")

            try:
                id = copilot.train(
                    question=question, sql=sql, ddl=ddl, documentation=documentation
                )

                return jsonify({"id": id})
            except Exception as e:
                print("TRAINING ERROR", e)
                return jsonify({"type": "error", "error": str(e)})

        @self.flask_app.route("/api/v0/create_function", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["question", "sql"])
        def create_function(user: any, id: str, question: str, sql: str):
            """
            Create function
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: function_template
                    id:
                      type: string
                    function_template:
                      type: object
            """
            plotly_code = self.cache.get(id=id, field="plotly_code")

            if plotly_code is None:
                plotly_code = ""

            function_data = self.copilot.create_function(question=question, sql=sql, plotly_code=plotly_code)

            return jsonify(
                {
                    "type": "function_template",
                    "id": id,
                    "function_template": function_data,
                }
            )

        @self.flask_app.route("/api/v0/update_function", methods=["POST"])
        @self.requires_auth
        def update_function(user: any):
            """
            Update function
            ---
            parameters:
              - name: user
                in: query
              - name: old_function_name
                in: body
                type: string
                required: true
              - name: updated_function
                in: body
                type: object
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
            """
            old_function_name = flask.request.json.get("old_function_name")
            updated_function = flask.request.json.get("updated_function")

            print("old_function_name", old_function_name)
            print("updated_function", updated_function)

            updated = copilot.update_function(old_function_name=old_function_name, updated_function=updated_function)

            return jsonify({"success": updated})

        @self.flask_app.route("/api/v0/delete_function", methods=["POST"])
        @self.requires_auth
        def delete_function(user: any):
            """
            Delete function
            ---
            parameters:
              - name: user
                in: query
              - name: function_name
                in: body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    success:
                      type: boolean
            """
            function_name = flask.request.json.get("function_name")

            return jsonify({"success": copilot.delete_function(function_name=function_name)})

        @self.flask_app.route("/api/v0/generate_followup_questions", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df", "question", "sql"])
        def generate_followup_questions(user: any, id: str, df, question, sql):
            """
            Generate followup questions
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: question_list
                    questions:
                      type: array
                      items:
                        type: string
                    header:
                      type: string
            """
            if self.allow_llm_to_see_data:
                followup_questions = copilot.generate_followup_questions(
                    question=question, sql=sql, df=df
                )
                if followup_questions is not None and len(followup_questions) > 5:
                    followup_questions = followup_questions[:5]

                self.cache.set(id=id, field="followup_questions", value=followup_questions)

                return jsonify(
                    {
                        "type": "question_list",
                        "id": id,
                        "questions": followup_questions,
                        "header": "Here are some potential followup questions:",
                    }
                )
            else:
                self.cache.set(id=id, field="followup_questions", value=[])
                return jsonify(
                    {
                        "type": "question_list",
                        "id": id,
                        "questions": [],
                        "header": "Followup Questions can be enabled if you set allow_llm_to_see_data=True",
                    }
                )

        @self.flask_app.route("/api/v0/generate_summary", methods=["GET"])
        @self.requires_auth
        @self.requires_cache(["df", "question"])
        def generate_summary(user: any, id: str, df, question):
            """
            Generate summary
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
            """
            if self.allow_llm_to_see_data:
                sql = self.cache.get(id=id, field="sql")
                summary = copilot.generate_summary(question=question, sql=sql, df=df)
                self.cache.set(id=id, field="summary", value=summary)

                return jsonify(
                    {
                        "type": "html",
                        "id": id,
                        "text": summary,
                    }
                )
            else:
                return jsonify(
                    {
                        "type": "text",
                        "id": id,
                        "text": "Can't summarize because the AI model doesn't have access to data",
                    }
                )

        @self.flask_app.route("/api/v0/load_question", methods=["GET"])
        @self.requires_auth
        def load_question(user: any):
            """
            Load question
            ---
            parameters:
              - name: user
                in: query
              - name: id
                in: query|body
                type: string
                required: true
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: question_cache
                    id:
                      type: string
                    question:
                      type: string
                    sql:
                      type: string
                    df:
                      type: object
                    fig:
                      type: object
                    summary:
                      type: string
            """
            user_id = flask.request.cookies.get("session_id")
            conv_id = flask.request.args.get("id", None)
            if not conv_id:
                return jsonify({"type": "error", "error": "`id` query-parameter is required"}), 400

            print("got this id:", conv_id)
            try:
                # return jsonify(
                #     {
                #         "type": "question_cache",
                #         "id": id,
                #         "question": question,
                #         "sql": sql,
                #         "df": df.head(10).to_json(orient="records", date_format="iso"),
                #         "fig": fig_json,
                #         "summary": summary,
                #     }
                # )
                return copilot.get_conversation_by_id(user_id=user_id, conversation_id=conv_id)

            except Exception as e:
                traceback.print_exc()
                return jsonify({"type": "error", "error": str(e)})

        @self.flask_app.route("/api/v0/get_question_history", methods=["GET"])
        @self.requires_auth
        def get_question_history(user: any):
            """
            Get question history
            ---
            parameters:
              - name: user
                in: query
            responses:
              200:
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: question_history
                    questions:
                      type: array
                      items:
                        type: string
            """

            user_id = flask.request.cookies.get("session_id")
            all_questions = cache.get_all(field_list=["question"])

            return jsonify(
                {
                    "type": "question_history",
                    "questions": copilot.get_conversation_history(user_id),
                }
            )

        @self.flask_app.route("/api/v0/<path:catch_all>", methods=["GET", "POST"])
        def catch_all(catch_all):
            return jsonify(
                {"type": "error", "error": "The rest of the API is not ported yet."}
            )

        if self.debug:
            @self.sock.route("/api/v0/log")
            def sock_log(ws):
                self.ws_clients.append(ws)

                try:
                    while True:
                        message = ws.receive()  # This example just reads and ignores to keep the socket open
                finally:
                    self.ws_clients.remove(ws)

    def run(self, *args, **kwargs):
        """
        Run the Flask app.

        Args:
            *args: Arguments to pass to Flask's run method.
            **kwargs: Keyword arguments to pass to Flask's run method.

        Returns:
            None
        """
        if args or kwargs:
            self.flask_app.run(*args, **kwargs)
        else:
            self.flask_app.secret_key = self.app_secret_key
            self.flask_app.run(host="0.0.0.0", port=8084, debug=self.debug, use_reloader=False)


class FlaskApp(FlaskAPI):
    def __init__(
            self,
            copilot: CopilotBase,
            cache: Cache = MemoryCache(),
            auth: AuthInterface = NoAuth(),
            debug=True,
            allow_llm_to_see_data=False,
            logo="",
            title="Welcome to Copilot",
            subtitle="Your AI-powered copilot for SQL queries.",
            show_training_data=True,
            suggested_questions=True,
            sql=True,
            table=True,
            csv_download=True,
            chart=True,
            redraw_chart=True,
            auto_fix_sql=True,
            ask_results_correct=True,
            followup_questions=True,
            summarization=True,
            function_generation=True,
            index_html_path=None,
            assets_folder=None,
            max_generated_question_length=200,
            max_attempts=3,
            app_secret_key=None
    ):
        """
        Expose a Flask app that can be used to interact with a Copilot instance.

        Args:
            copilot: The Copilot instance to interact with.
            cache: The cache to use. Defaults to MemoryCache, which uses an in-memory cache. You can also pass in a custom cache that implements the Cache interface.
            auth: The authentication method to use. Defaults to NoAuth, which doesn't require authentication. You can also pass in a custom authentication method that implements the AuthInterface interface.
            debug: Show the debug console. Defaults to True.
            allow_llm_to_see_data: Whether to allow the LLM to see data. Defaults to False.
            logo: The logo to display in the UI.
            title: The title to display in the UI. Defaults to "Welcome to Copilot".
            subtitle: The subtitle to display in the UI. Defaults to "Your AI-powered copilot for SQL queries.".
            show_training_data: Whether to show the training data in the UI. Defaults to True.
            suggested_questions: Whether to show suggested questions in the UI. Defaults to True.
            sql: Whether to show the SQL input in the UI. Defaults to True.
            table: Whether to show the table output in the UI. Defaults to True.
            csv_download: Whether to allow downloading the table output as a CSV file. Defaults to True.
            chart: Whether to show the chart output in the UI. Defaults to True.
            redraw_chart: Whether to allow redrawing the chart. Defaults to True.
            auto_fix_sql: Whether to allow auto-fixing SQL errors. Defaults to True.
            ask_results_correct: Whether to ask the user if the results are correct. Defaults to True.
            followup_questions: Whether to show followup questions. Defaults to True.
            summarization: Whether to show summarization. Defaults to True.
            index_html_path: Path to the index.html. Defaults to None, which will use the default index.html
            assets_folder: The location where you'd like to serve the static assets from. Defaults to None, which will use hardcoded Python variables.

        Returns:
            None
        """
        super().__init__(copilot, cache, auth, debug, allow_llm_to_see_data, chart, max_generated_question_length,
                         max_attempts, app_secret_key)

        self.config["logo"] = logo
        self.config["title"] = title
        self.config["subtitle"] = subtitle
        self.config["show_training_data"] = show_training_data
        self.config["suggested_questions"] = suggested_questions
        self.config["sql"] = sql
        self.config["table"] = table
        self.config["csv_download"] = csv_download
        self.config["chart"] = chart
        self.config["redraw_chart"] = redraw_chart
        self.config["auto_fix_sql"] = auto_fix_sql
        self.config["ask_results_correct"] = ask_results_correct
        self.config["followup_questions"] = followup_questions
        self.config["summarization"] = summarization
        self.config["function_generation"] = function_generation and hasattr(copilot, "get_function")
        self.config["version"] = "0.5"

        self.index_html_path = index_html_path
        self.assets_folder = assets_folder

        @self.flask_app.route("/auth/login", methods=["POST"])
        def login():
            return self.auth.login_handler(flask.request)

        @self.flask_app.route("/auth/callback", methods=["GET"])
        def callback():
            return self.auth.callback_handler(flask.request)

        @self.flask_app.route("/auth/logout", methods=["GET"])
        def logout():
            return self.auth.logout_handler(flask.request)

        @self.flask_app.route("/assets/<path:filename>")
        def proxy_assets(filename):
            if self.assets_folder:
                return send_from_directory(self.assets_folder, filename)

            if ".css" in filename:
                return Response(css_content, mimetype="text/css")

            if ".js" in filename:
                return Response(js_content, mimetype="text/javascript")

            # Return 404
            return "File not found", 404

        @self.flask_app.route("/", defaults={"path": ""})
        @self.flask_app.route("/<path:path>")
        def hello(path: str):

            current_session_id = flask.request.cookies.get("session_id")
            generated_session_id = current_session_id if current_session_id else hashlib.sha1(
                str(time.time_ns()).encode()).hexdigest()

            if self.index_html_path:
                directory = os.path.dirname(self.index_html_path)
                filename = os.path.basename(self.index_html_path)
                response = make_response(send_from_directory(directory=directory, path=filename))
                response.set_cookie('session_id', generated_session_id)
                return response

            response = make_response(html_content)
            response.set_cookie('session_id', generated_session_id)
            return response

    def test(self, ):
        return self.get_related_ddl("test")
