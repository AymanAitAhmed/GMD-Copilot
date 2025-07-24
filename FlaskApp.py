import json
import logging
import traceback
import os
import sys
import uuid
import hashlib
import time
import threading
import queue
from abc import ABC, abstractmethod
from functools import wraps
import importlib.metadata
import markdown
import re
import sqlite3

import pandas as pd
import duckdb
import flask
from flask_cors import CORS
import requests
from flasgger import Swagger
from flask import Flask, Response, jsonify, request, send_from_directory, session, make_response, flash, g
from werkzeug.utils import secure_filename
from flask_sock import Sock

from webApp.base.base import CopilotBase
from webApp.assets import *
from webApp.auth import AuthInterface, NoAuth
from webApp.base.cache_types import CacheTypes
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
    def get_all(self) -> list:
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

    def get_all(self):
        print(self.cache)
        print(self.cache.keys())
        print(self.cache.values())
        return jsonify({'content': "self.cache"})

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

        # Register CORS for your API
        CORS(
            self.flask_app,
            supports_credentials=True,
            resources={r"/api/*": {"origins": ["http://192.168.7.99:3001", 'http://localhost:3001']}}
        )

        # Setup teardown for request‑scoped DB
        @self.flask_app.teardown_appcontext
        def close_db_conn(exc):
            conn = g.pop('db_conn', None)
            if conn:
                conn.close()

            chart_conn = g.pop('chart_db_conn', None)
            if chart_conn:
                chart_conn.close()

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

        @self.flask_app.route("/api/v0/get_cache", methods=["GET"])
        def get_cache():
            return self.cache.get_all()

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
            print("start")
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
                print("error: ", e)
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
            the session ID stored in Flask cookies.

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

        @self.flask_app.route("/api/v0/create_conversation_contextual", methods=["POST"])
        def create_conversation_contextual():
            """
            Create a new conversation with contextual tool information.
            ---
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    user_question:
                      type: string
                      description: The initial question from the user.
                    user_id:
                      type: string
                      description: The ID of the user.
            responses:
              200:
                description: Conversation created successfully
                schema:
                  type: object
                  properties:
                    type:
                      type: string
                      default: success
                    conversation_id:
                      type: string
              400:
                description: Bad request (e.g., missing user_question or user_id)
              500:
                description: Internal server error
            """
            if not request.json:
                print("Request body must be JSON for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "Request body must be JSON"}), 400

            user_question = request.json.get("user_question")
            print(request.json)
            user_id = request.json.get("user_id", 'empty')
            conversation_id = request.json.get("conversation_id")

            if not user_question:
                print("Missing user_question in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "user_question is required"}), 400
            if not user_id:
                print("Missing user_id in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "user_id is required"}), 400

            try:
                dfs = get_uploaded_spreadsheets()
                # Call the modified create_new_conversation method from CopilotBase instance
                conversation_id = self.copilot.create_new_conversation(user_question=user_question,
                                                                       user_id=str(user_id),
                                                                       dfs=dfs
                                                                       )
                print(f"Successfully created conversation_id: {conversation_id} via /create_conversation_contextual")
                return jsonify({"type": "success", "conversation_id": conversation_id}), 200
            except Exception as e:
                # Ensure traceback is imported in FlaskApp.py if not already
                tb_str = traceback.format_exc()
                print(f"Error in /create_conversation_contextual: {e}\n{tb_str}", "Error")
                return jsonify({"type": "error", "error": "Failed to create conversation", "details": str(e)}), 500

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

            if not request.json:
                print("Request body must be JSON for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "Request body must be JSON"}), 400

            conversation_id = request.json.get("conversation_id")
            question = request.json.get("question")
            user_id = request.json.get("user_id", "empty")

            if not conversation_id:
                print("Missing conversation_id in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "conversation_id is required"}), 400

            if not user_id:
                print("Missing user_id in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "user_id is required"}), 400

            sql = self.copilot.generate_sql(user_id=user_id, conversation_id=conversation_id, )

            self.cache.set(id=conversation_id, field=CacheTypes.QUESTION, value=question)
            self.cache.set(id=conversation_id, field=CacheTypes.SQL, value=sql)

            if self.copilot.is_sql_valid(sql=sql):
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

            self.cache.set(id=id, field=CacheTypes.QUESTION, value=question)
            self.cache.set(id=id, field=CacheTypes.SQL, value=function['instantiated_sql'])

            if 'instantiated_post_processing_code' in function and function[
                'instantiated_post_processing_code'] is not None and len(
                function['instantiated_post_processing_code']) > 0:
                self.cache.set(id=id, field=CacheTypes.PLOTLY_CODE, value=function['instantiated_post_processing_code'])

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

        @self.flask_app.route("/api/v0/run_sql", methods=["POST"])
        def run_sql():

            if not request.json:
                print("Request body must be JSON for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "Request body must be JSON"}), 400

            conversation_id = request.json.get("conversation_id")
            sql = request.json.get("sql")
            allow_correction = request.json.get("allow_correction",True)
            user_id = request.cookies.get("session_id", "empty")
            attempt = 0
            last_error = None

            if not conversation_id:
                print("Missing conversation_id in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "conversation_id is required"}), 400
            # if not user_id:
            #     print("Missing user_id in request for /create_conversation_contextual", "Error")
            #     return jsonify({"type": "error", "error": "user_id is required"}), 400
            if not sql:
                print("Missing sql in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "sql is required"}), 400

            if not self.copilot.run_sql_is_set:
                return jsonify({
                    "type": "error",
                    "error": "Please connect to a database using copilot.connect_to_... in order to run SQL queries."
                })

            dfs = get_uploaded_spreadsheets()
            if len(dfs) >= 1:
                for i, df in enumerate(dfs):
                    self.copilot.con.sql(f"""CREATE OR REPLACE TEMP TABLE table{i} AS SELECT * FROM df;""")

            while attempt < self.max_attempts:
                try:
                    df_returned = self.copilot.con.sql(query=sql).df()

                    self.cache.set(id=conversation_id, field=CacheTypes.DF, value=df_returned)
                    self.cache.set(id=conversation_id, field=CacheTypes.SQL, value=sql)

                    return jsonify({
                        "type": "df",
                        "id": conversation_id,
                        "fixing_attempts": attempt,
                        "df": df_returned.head(100).to_json(orient='records', date_format='iso'),
                        "should_generate_chart": self.chart and copilot.should_generate_chart(df_returned),
                    })
                except Exception as e:
                    last_error = str(e)
                    attempt += 1

                    if attempt < self.max_attempts:
                        original_question = self.cache.get(conversation_id, CacheTypes.QUESTION)
                        fix_question = f"I have an error: {last_error}\n\nHere is the SQL I tried to run: {sql}\n\nThis is the question I was trying to answer: {original_question}\n\nCan you rewrite the SQL to fix the error?"

                        print(f"error in Generated SQL, Fixing the SQL, Attempt:{attempt}/{self.max_attempts}")
                        sql = self.copilot.generate_sql(conversation_id=conversation_id, user_id=user_id)

                        self.cache.set(id=conversation_id, field=CacheTypes.SQL, value=sql)
                    else:
                        print(traceback.format_exc())
                        return jsonify({"type": "sql_error",
                                        "error": f"The Copilot tried fixing the error for {max_attempts} times but didn't succeed."})
                finally:
                    self.copilot.insert_message_db(conversation_id, user_id, "assistant", extracted_sql=sql)

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

            self.cache.set(id=id, field=CacheTypes.SQL, value=fixed_sql)

            return jsonify(
                {
                    "type": "sql",
                    "id": id,
                    "text": fixed_sql,
                }
            )

        @self.flask_app.route('/api/v0/update_sql', methods=['POST'])
        def update_sql():
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

            if not request.json:
                print("Request body must be JSON for /update_sql", "Error")
                return jsonify({"type": "error", "error": "Request body must be JSON"}), 400

            sql = flask.request.json.get('sql')
            conversation_id = flask.request.json.get('conversation_id')

            if not conversation_id:
                print("Missing conversation_id in request for /update_sql", "Error")
                return jsonify({"type": "error", "error": "conversation_id is required"}), 400

            if not sql:
                print("Missing sql in request for /update_sql", "Error")
                return jsonify({"type": "error", "error": "sql is required"}), 400

            self.cache.set(id=conversation_id, field=CacheTypes.SQL, value=sql)

            return jsonify(
                {
                    "type": "sql",
                    "id": conversation_id,
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

        @self.flask_app.route("/api/v0/generate_vega_chart", methods=["GET"])
        def generate_vega_chart():

            user_id = flask.request.cookies.get("session_id")

            if not user_id:
                return jsonify({
                    'type': 'error',
                    'error': "please provide the user_id"
                })

            question = flask.request.args.get("question", None)
            if not question:
                return jsonify({
                    'type': 'error',
                    'error': "please provide the question asked by the user."
                })

            sql = flask.request.args.get("sql", None)
            if not sql:
                return jsonify({
                    'type': 'error',
                    'error': "please provide the SQL query generated."
                })
            # chart_instructions = flask.request.args.get('chart_instructions')
            df = self.copilot.con.sql(sql).df()
            try:
                # # Try any shortcut first (if you have common‐vega snippets)
                # code = copilot.generate_common_vega(df)
                # if not code:
                #     # build LLM prompt if no common snippet or user overrides
                #     instr = f". When generating the chart, use these special instructions: {chart_instructions}" if chart_instructions else ""
                #     prompt_question = f"{question}{instr}"

                # Execute and extract the Altair Chart object
                spec = self.copilot.get_vega_spec(question, sql, df)

                self.cache.set(id=user_id, field=CacheTypes.SPEC_JSON, value=spec)

                return jsonify({
                    "type": "vega_spec",
                    "spec": spec,
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"type": "error", "error": str(e)})

        @self.flask_app.route("/api/v0/generate_plotly_figure", methods=["POST"])
        def generate_plotly_figure():

            if not request.json:
                print("Request body must be JSON for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "Request body must be JSON"}), 400

            user_id = request.cookies.get("session_id", "empty")
            conversation_id = request.json.get("conversation_id")
            sql = request.json.get("sql")
            question = flask.request.json.get("question", None)

            if not user_id:
                print("Missing user_id in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "user_id is required"}), 400
            if not conversation_id:
                print("Missing conversation_id in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "conversation_id is required"}), 400
            if not sql:
                print("Missing sql in request for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "sql is required"}), 400
            if not question:
                return jsonify({'type': 'error', 'error': "please provide the question asked by the user."}), 400

            fig_json = self.cache.get(id=conversation_id, field=CacheTypes.FIG_JSON)
            chart_instructions = request.json.get('chart_instructions')
            # if fig_json and not chart_instructions:
            #     return jsonify(
            #         {
            #             "type": "plotly_figure",
            #             "fig": fig_json,
            #         }
            #     )

            print("chart instructions:", chart_instructions)
            df = self.cache.get(conversation_id, CacheTypes.DF)
            try:
                code = ""
                if chart_instructions is None or len(chart_instructions) == 0:
                    code = self.copilot.generate_common_plotly(df)

                if code == "" and (chart_instructions is None or len(chart_instructions) == 0):
                    code = copilot.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                        df_sample=df.head(5)
                    )
                    self.cache.set(id=id, field=CacheTypes.PLOTLY_CODE, value=code)
                elif code == "":
                    question = f"{question}. When generating the chart, use these special instructions: {chart_instructions}"
                    code = copilot.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                        df_sample=df.head(5)
                    )
                    self.cache.set(id=id, field=CacheTypes.PLOTLY_CODE, value=code)

                fig = copilot.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
                fig_json = fig.to_json()
                print("the plotly json:", fig_json)
                self.cache.set(id=conversation_id, field=CacheTypes.FIG_JSON, value=fig_json)

                return jsonify(
                    {
                        "type": "plotly_figure",
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
            df = self.copilot.get_training_data()
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

        @self.flask_app.route("/api/v0/api/stream_answer", methods=["GET"])
        def stream_answer():

            user_id = request.cookies.get("session_id", "empty")
            conversation_id = request.args.get("conversation_id")

            if not user_id:
                print("Missing user_id in request for /handle_streaming", "Error")
                return jsonify({"type": "error", "error": "user_id is required"}), 400
            if not conversation_id:
                print("Missing conversation_id in request for /handle_streaming", "Error")
                return jsonify({"type": "error", "error": "conversation_id is required"}), 400

            summary_answer = self.cache.get(conversation_id, CacheTypes.SUMMARY_ANSWER)
            sql = self.cache.get(conversation_id, CacheTypes.SQL)
            question = self.cache.get(conversation_id, CacheTypes.QUESTION)
            df = self.cache.get(conversation_id, CacheTypes.DF)

            if summary_answer:
                def generate_cached():
                    data = {'content': summary_answer, 'done': True}
                    yield f"data: {json.dumps(data)}\n\n"

                return Response(generate_cached(), mimetype='text/event-stream')

            def generate_stream_response():
                try:
                    for chunk_content in self.copilot.generate_summary(question=question, sql=sql, df=df):
                        response_chunk = f"data: {json.dumps({'content': chunk_content, 'done': False})}\n\n"
                        yield response_chunk
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    self.cache.set(id=conversation_id, field=CacheTypes.SUMMARY_ANSWER, value=None)
                except ValueError as ve:  # Catch prompt validation errors specifically
                    print(f"Validation error for streaming request: {ve}")
                    yield f"data: {json.dumps({'error': str(ve), 'done': True})}\n\n"
                except Exception as e:
                    print(f"Error during streaming generation: {e}")
                    yield f"data: {json.dumps({'error': 'An error occurred during streaming.', 'done': True})}\n\n"

            return Response(generate_stream_response(), mimetype='text/event-stream')

        # @self.flask_app.route("/api/v0/run_agent", methods=["POST"])
        # def run_agent_endpoint() -> any:
        #     data = request.get_json()
        #     question = data.get("question")
        #     if not question:
        #         return jsonify({"error": "Missing 'question' in request body."}), 400
        #
        #     user_id = "userid_12335"
        #     thread_id = data.get("thread_id") or None
        #     print("cookies: ", flask.request.cookies.to_dict())
        #     print(f"run_agent: question='{question}', user_id='{user_id}', thread_id='{thread_id}'")
        #
        #     messages, conversation_id = self.copilot.create_or_load_conversation(thread_id, user_id, question)
        #     print(f"run_agent: Created conversation_id='{conversation_id}'")
        #
        #     # output_queue = queue.Queue()
        #     #
        #     # def stream_worker(q_out, qst, usr_id, conv_id):
        #     #     worker_log_prefix = f"Worker {conv_id}:"
        #     #     print(f"{worker_log_prefix} Starting.")
        #     #     try:
        #     #         # cancel_requested flag is initialized by run_agent_endpoint
        #     #
        #     #         for response_item in self.copilot.stream_response(qst, usr_id, conv_id):
        #     #             if self.cache.get(conv_id, "cancel_requested"):
        #     #                 cached_status_on_cancel_detect = self.cache.get(conv_id, 'status')
        #     #                 print(
        #     #                     f"{worker_log_prefix} Detected cancel_requested. Current status in cache: {cached_status_on_cancel_detect}")
        #     #                 self.cache.set(conv_id, "status", "CANCELLED")  # Set status FIRST
        #     #                 print(f"{worker_log_prefix} Set status to CANCELLED in cache.")
        #     #                 q_out.put(json.dumps({
        #     #                     "message": "Task cancelled by user.",
        #     #                     "done": True,
        #     #                     "cancelled": True,
        #     #                     "queryId": conv_id
        #     #                 }))
        #     #                 q_out.put(None)
        #     #                 print(f"{worker_log_prefix} Signalled cancellation and end of stream.")
        #     #                 return
        #     #
        #     #             q_out.put(response_item)
        #     #
        #     #         # If loop completes without cancellation
        #     #         if not self.cache.get(conv_id, "cancel_requested"):
        #     #             print(f"{worker_log_prefix} Natural completion of stream_response. Signalling end.")
        #     #             # The streaming_endpoint or last message should handle setting FINISHED status.
        #     #             # Worker ensures None is sent.
        #     #             q_out.put(None)
        #     #
        #     #     except Exception as e:
        #     #         if self.cache.get(conv_id, "cancel_requested"):
        #     #             print(
        #     #                 f"{worker_log_prefix} Error during cancellation process: {e}. Status should already be CANCELLED or CANCELLING.")
        #     #             if self.cache.get(conv_id, "status") != "CANCELLED":
        #     #                 self.cache.set(conv_id, "status", "CANCELLED")
        #     #             q_out.put(json.dumps({
        #     #                 "message": f"Task cancelled, error during shutdown: {e}",
        #     #                 "done": True, "cancelled": True, "error": True, "queryId": conv_id
        #     #             }))
        #     #         else:
        #     #             print(f"{worker_log_prefix} Error: {e}")
        #     #             traceback.print_exc()
        #     #             self.cache.set(conv_id, "status", "ERROR")
        #     #             self.cache.set(conv_id, "error_message", str(e))
        #     #             q_out.put(json.dumps({
        #     #                 "message": f"An error occurred: {e}", "done": True, "error": True, "queryId": conv_id
        #     #             }))
        #     #         q_out.put(None)  # Ensure stream termination signal
        #     #     finally:
        #     #         print(f"{worker_log_prefix} Exiting. Final Status in cache: {self.cache.get(conv_id, 'status')}")
        #     #
        #     # # Before starting thread:
        #     # self.cache.set(conversation_id, "status", "STREAMING")
        #     # self.cache.set(conversation_id, "type", "TEXT_TO_SQL")
        #     # self.cache.set(conversation_id, "question", question)
        #     # self.cache.set(conversation_id, "user_id", user_id)
        #     # self.cache.set(conversation_id, "thread_id", thread_id)
        #     # self.cache.set(conversation_id, "output_queue", output_queue)
        #     # self.cache.set(conversation_id, "cancel_requested", False)  # Initialize the flag
        #     #
        #     # worker_thread = threading.Thread(
        #     #     target=stream_worker,
        #     #     args=(output_queue, question, user_id, conversation_id),
        #     #     daemon=True
        #     # )
        #     # worker_thread.start()
        #     # print(f"run_agent: Worker thread started for conversation_id='{conversation_id}'")
        #
        #     return jsonify({
        #         "askingTask": {
        #             "queryId": conversation_id, "thread_id": conversation_id, "status": "STREAMING",
        #             "type": "TEXT_TO_SQL", "question": question,
        #             "retrievedTables": [], "sqlGenerationReasoning": ""
        #         }
        #     })
        #
        # @self.flask_app.route("/api/v0/api/ask_task/streaming", methods=["GET"])
        # def streaming_endpoint():
        #     conversation_id = request.args.get('query_id')
        #     stream_log_prefix = f"StreamEP {conversation_id}:"
        #
        #     if not conversation_id:
        #         return jsonify({"error": "Missing queryId parameter"}), 400
        #
        #     output_queue = self.cache.get(conversation_id, "output_queue")
        #
        #     if not output_queue:
        #         status = self.cache.get(conversation_id, "status")
        #         if status == "CANCELLED":
        #             return jsonify(
        #                 {"message": "Task was cancelled.", "queryId": conversation_id, "status": "CANCELLED"}), 410
        #         elif status == "ERROR":
        #             error_msg = self.cache.get(conversation_id, "error_message") or "Task errored before streaming."
        #             return jsonify({"error": error_msg, "queryId": conversation_id, "status": "ERROR"}), 500
        #         return jsonify({"error": "Invalid queryId or task not found/started"}), 404
        #
        #     def generate():
        #         while True:
        #             try:
        #                 response = output_queue.get(timeout=1)
        #             except queue.Empty:
        #                 cached_status = self.cache.get(conversation_id, "status")
        #                 cancel_flag = self.cache.get(conversation_id, "cancel_requested")
        #
        #                 if cached_status in ["CANCELLED", "ERROR", "FINISHED"]:
        #                     print(
        #                         f"{stream_log_prefix} Ending (timeout) as task status in cache is terminal: {cached_status}.")
        #                     break
        #                 elif cached_status == "CANCELLING" or (cached_status == "STREAMING" and cancel_flag):
        #                     continue
        #                 else:  # STREAMING without cancel flag, or PENDING etc.
        #                     continue
        #
        #             if response is None:
        #                 cached_status_at_none = self.cache.get(conversation_id, "status")
        #                 cancel_flag_at_none = self.cache.get(conversation_id, "cancel_requested")
        #                 error_msg_at_none = self.cache.get(conversation_id, "error_message")
        #                 # Fallback if worker died without setting a clear final status
        #                 if cached_status_at_none in ["STREAMING", "CANCELLING", None]:  # None status means not set
        #                     if cancel_flag_at_none:
        #                         if cached_status_at_none != "CANCELLED":
        #                             self.cache.set(conversation_id, "status", "CANCELLED")
        #                             print(
        #                                 f"{stream_log_prefix} Status was ambiguous on None, set to CANCELLED due to request flag.")
        #                     elif error_msg_at_none:  # If an error was logged by worker
        #                         if cached_status_at_none != "ERROR":
        #                             self.cache.set(conversation_id, "status", "ERROR")
        #                             print(
        #                                 f"{stream_log_prefix} Status was ambiguous on None, set to ERROR due to error_message.")
        #                     elif cached_status_at_none != "FINISHED":  # Default to FINISHED if no cancel/error
        #                         self.cache.set(conversation_id, "status", "FINISHED")
        #                         print(f"{stream_log_prefix} Status was ambiguous on None, set to FINISHED.")
        #                 break  # Exit generate() loop
        #
        #             yield f"data: {response}\n\n"
        #
        #             try:
        #                 response_data = json.loads(response)
        #                 is_done = response_data.get('done')
        #                 is_error = response_data.get('error', False)
        #                 is_cancelled_msg = response_data.get('cancelled', False)
        #
        #                 if is_cancelled_msg:
        #                     if self.cache.get(conversation_id, "status") != "CANCELLED":
        #                         self.cache.set(conversation_id, "status", "CANCELLED")
        #
        #                 elif is_done:
        #                     status_before_done = self.cache.get(conversation_id, "status")
        #                     cancel_pending = self.cache.get(conversation_id, "cancel_requested") or \
        #                                      status_before_done == "CANCELLING"
        #                     print(
        #                         f"{stream_log_prefix} Received 'done:true' message. Cancel pending: {cancel_pending}, Status before: {status_before_done}")
        #
        #                     if cancel_pending:
        #                         print(
        #                             f"{stream_log_prefix} 'done:true' but cancel is pending. Deferring final status to worker/None signal.")
        #                     elif status_before_done not in ["CANCELLED", "ERROR", "FINISHED"]:
        #                         if is_error:
        #                             self.cache.set(conversation_id, "status", "ERROR")
        #                             if not self.cache.get(conversation_id, "error_message"):
        #                                 self.cache.set(conversation_id, "error_message",
        #                                                response_data.get("message", "Error from stream"))
        #                             print(f"{stream_log_prefix} Set status to ERROR via done:true,error:true message.")
        #                         else:
        #                             self.cache.set(conversation_id, "status", "FINISHED")
        #                             if response_data.get('sql'): self.cache.set(conversation_id, "sql",
        #                                                                         response_data.get('sql'))
        #                             print(f"{stream_log_prefix} Set status to FINISHED via done:true message.")
        #             except json.JSONDecodeError:
        #                 print(f"{stream_log_prefix} Warning: Could not parse JSON from stream: {response}")
        #             except Exception as e:
        #                 print(f"{stream_log_prefix} Error processing streamed message: {e}")
        #                 if self.cache.get(conversation_id, "status") not in ["CANCELLED", "ERROR", "FINISHED"]:
        #                     self.cache.set(conversation_id, "status", "ERROR")
        #                     self.cache.set(conversation_id, "error_message", f"Error processing stream: {e}")
        #
        #         final_cached_status = self.cache.get(conversation_id, "status")
        #
        #     return Response(generate(), mimetype='text/event-stream')

        @self.flask_app.route("/api/v0/api/ask_task/status", methods=["GET"])
        def task_status_endpoint():
            query_id = request.args.get('query_id')
            if not query_id:
                return jsonify({"error": "Missing queryId parameter"}), 400

            # Retrieve task info from cache
            status = self.cache.get(query_id, CacheTypes.STATUS)
            task_type = self.cache.get(query_id, "type") or "TEXT_TO_SQL"
            question = self.cache.get(query_id, CacheTypes.QUESTION) or ""

            if status is None:
                if self.cache.get(query_id, "output_queue"):
                    status = "PENDING"
                else:
                    return jsonify({"error": "Task with specified queryId not found or not initialized.",
                                    "queryId": query_id}), 404

            response_task_data = {
                "queryId": query_id, "status": status, "type": task_type, "question": question,
                "retrievedTables": self.cache.get(query_id, "retrievedTables") or [],
                "sqlGenerationReasoning": self.cache.get(query_id, "sqlGenerationReasoning") or ""
            }

            if status == "FINISHED":
                response_task_data["sql"] = self.cache.get(query_id, CacheTypes.SQL)
            elif status == "ERROR":
                response_task_data["error_message"] = self.cache.get(query_id, "error_message") or "Unspecified error."
            elif status == "CANCELLED":
                response_task_data["message"] = "Task was cancelled by the user."

            return jsonify({"askingTask": response_task_data})

        @self.flask_app.route("/api/v0/api/ask_task/cancel", methods=["POST"])
        def cancel_task_endpoint() -> any:
            data = request.get_json()
            query_id = data.get("query_id")
            if not query_id:
                return jsonify({"error": "Missing 'queryId' in request body."}), 400

            cancel_log_prefix = f"CancelEP {query_id}:"
            print(f"{cancel_log_prefix} Request received.")

            current_status = self.cache.get(query_id, CacheTypes.STATUS)
            output_queue_exists = bool(self.cache.get(query_id, "output_queue"))
            print(f"{cancel_log_prefix} Current status: {current_status}, Output queue exists: {output_queue_exists}")

            if not output_queue_exists and current_status not in ["STREAMING", "CANCELLING",
                                                                  "PENDING"]:  # Check if it ever really started
                print(f"{cancel_log_prefix} Task not found or never fully initialized for cancellation.")
                return jsonify({"message": "Task not found or never fully initialized.", "queryId": query_id}), 404

            if current_status in ["FINISHED", "ERROR", "CANCELLED"]:
                print(f"{cancel_log_prefix} Task already in terminal state: {current_status}.")
                return jsonify({"message": f"Task {query_id} is already {current_status}."}), 200
            if current_status == "CANCELLING":
                print(f"{cancel_log_prefix} Task already being cancelled.")
                return jsonify({"message": f"Task {query_id} is already being cancelled."}), 200

            self.cache.set(query_id, "cancel_requested", True)
            self.cache.set(query_id, "status", "CANCELLING")
            print(f"{cancel_log_prefix} Set cancel_requested=True, status=CANCELLING.")

            return jsonify({"message": "Cancellation request processed. Task is being stopped.", "queryId": query_id,
                            "status": "CANCELLING"}), 200

        @self.flask_app.route("/api/v0/train", methods=["POST"])
        def add_training_data():
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

            if not request.json:
                print("Request body must be JSON for /create_conversation_contextual", "Error")
                return jsonify({"type": "error", "error": "Request body must be JSON"}), 400

            question = flask.request.json.get("question")
            sql = flask.request.json.get("sql")
            # ddl = flask.request.json.get("ddl")
            # documentation = flask.request.json.get("documentation")
            # conversation_id = request.json.get("conversation_id")
            # user_id = request.cookies.get("session_id", "empty")

            if not question:
                print("Missing question in request for /add_training_data", "Error")
                return jsonify({"type": "error", "error": "conversation_id is required"}), 400

            if not sql:
                print("Missing sql in request for /add_training_data", "Error")
                return jsonify({"type": "error", "error": "sql is required"}), 400

            try:
                id = self.copilot.train(question=question, sql=sql)

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
            plotly_code = self.cache.get(id=id, field=CacheTypes.PLOTLY_CODE)

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
                sql = self.cache.get(id=id, field=CacheTypes.SQL)
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

                conversation = self.copilot.get_conversation_by_id(user_id=user_id, conversation_id=conv_id)
                print("conversation:", conversation)
                return conversation

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

            history_list = self.copilot.get_conversation_history(user_id)
            return jsonify(
                {
                    "type": "question_history",
                    "questions": history_list,
                }
            )

        @self.flask_app.route("/api/v0/delete_conversation", methods=["DELETE"])
        @self.requires_auth
        def delete_conversation(user: any):

            user_id = flask.request.cookies.get("session_id")
            conv_id = flask.request.args.get("id", None)
            if not conv_id:
                return jsonify({"type": "error", "error": "`id` query-parameter is required"}), 400

            result = self.copilot.delete_conversation(user_id,conv_id)
            if result:
                return jsonify({"type": "message", "content": "Conversation was deleted successfully"}), 200
            else:
                return jsonify({"type": "error", "error": "Couldn't delete the conversation"}), 500

        # -----------------------Dashboard-----------------------------
        @self.flask_app.route("/api/add_chart", methods=["POST"])
        @self.requires_auth
        def add_chart(user):
            """
            Adds a new chart to the database.
            ---
            parameters:
              - name: body
                in: body
                required: true
                schema:
                  type: object
                  properties:
                    chart_json:
                      type: object
            responses:
              200:
                description: Chart added successfully
              400:
                description: Missing chart_json in request body
              500:
                description: Failed to add chart
            """

            user_id = flask.request.cookies.get("session_id")
            conv_id = flask.request.args.get("id", None)
            if not conv_id:
                return jsonify({"type": "error", "error": "`id` query-parameter is required"}), 400

            chart_json_str = self.cache.get(conv_id,CacheTypes.FIG_JSON)
            if chart_json_str is None:
                if not request.json or 'chart_json' not in request.json:
                    return jsonify({"error": "Missing chart_json in request body"}), 400

                chart_json_str = json.dumps(request.json['chart_json'])

            try:
                chart_id = self.copilot.add_chart(chart_json_str)
                return jsonfy({"success": True, "chart_id": chart_id})
            except Exception as e:
                self.copilot.log(f"Error adding chart: {e}", "Error")
                return jsonify({"error": "Failed to add chart"}), 500

        @self.flask_app.route("/api/get_charts", methods=["GET"])
        @self.requires_auth
        def get_charts(user):
            """
            Retrieves all charts from the database.
            ---
            responses:
              200:
                description: A list of charts
              500:
                description: Failed to retrieve charts
            """
            try:
                charts = self.copilot.get_all_charts()
                return jsonify({"charts": charts})
            except Exception as e:
                self.copilot.log(f"Error getting charts: {e}", "Error")
                return jsonify({"error": "Failed to retrieve charts"}), 500


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
        # Ensure DB table exists and enable WAL before starting
        CONVERSATION_DB_PATH = r"./webApp/data/conversations.db"

        with sqlite3.connect(CONVERSATION_DB_PATH) as init_conn:
            init_conn.execute("PRAGMA journal_mode=WAL;")
            init_conn.execute("PRAGMA synchronous=NORMAL;")
            init_conn.execute(
                '''
                CREATE TABLE IF NOT EXISTS message_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    tool_call_id TEXT,
                    content TEXT,
                    tool_calls TEXT,
                    extracted_sql TEXT,
                    reasoning TEXT
                );
                '''
            )
            init_conn.commit()

        # Run the Flask server
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
            function_generation: Whether to show function generation. Defaults to True.
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
