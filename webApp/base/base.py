import json
import os
import re
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from urllib.parse import urlparse
import markdown
import ast
import pprint
import sqlite3
import datetime
import threading

import pandas as pd
import duckdb
import io
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlparse

from .exceptions import ImproperlyConfigured, ValidationError
from .training_plan import TrainingPlan, TrainingPlanItem

CONVERSATION_DB_PATH = r"C:\Users\ACER\PycharmProjects\LogisticRegression\env\text2sql\webApp\data\conversations.db"
thread_local = threading.local()


class CopilotBase(ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)

    def log(self, message: str, title: str = "Info"):
        print(f"{title}: {message}")

    def _response_language(self) -> str:
        if self.language is None:
            return ""

        return f"You must Respond {self.language} language."

    # ----database related----

    # def _initialize_database(self):
    #     """Initializes the SQLite database connection and creates the table if needed."""
    #     try:
    #         self.conn = sqlite3.connect(self.db_path)
    #         self.cursor = self.conn.cursor()
    #         self.cursor.execute("""
    #              CREATE TABLE IF NOT EXISTS message_history (
    #                  timestamp DATETIME NOT NULL,
    #                  conversation_id TEXT PRIMARY KEY NOT NULL,
    #                  user_id TEXT NOT NULL,
    #                  role TEXT NOT NULL,
    #                  content TEXT NOT NULL,
    #                  tool_call_id TEXT, -- Optional: For linking tool requests and responses
    #                  tool_name TEXT     -- Optional: Name of the tool called/responded
    #              )
    #          """)
    #         self.conn.commit()
    #         print(f"Database initialized successfully at {self.db_path}")
    #     except sqlite3.Error as e:
    #         print(f"Database error during initialization: {e}")
    #         # Potentially raise the error or handle it more gracefully
    #         if self.conn:
    #             self.conn.close()
    #
    # def close_db(self):
    #     """Closes the database connection."""
    #     if self.conn:
    #         self.conn.close()
    #         print("Database connection closed.")

    def get_connection(self, db_path=CONVERSATION_DB_PATH):
        conn = getattr(thread_local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS message_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_call_id TEXT,
                    tool_name TEXT,
                    extracted_sql TEXT,
                    reasoning TEXT
                );
            ''')
            conn.commit()
            thread_local.conn = conn
        return conn

    def insert_message_db(self, conversation_id, user_id, role, content=None, tool_call_id=None, tool_name=None,
                          extracted_sql=None, reasoning=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
        cursor.execute(
            """INSERT INTO message_history (timestamp,conversation_id,user_id,role,content,tool_call_id,tool_name,extracted_sql,reasoning)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (timestamp, conversation_id, user_id, role, content, tool_call_id, tool_name, extracted_sql, reasoning)
        )
        conn.commit()
        cursor.close()

    def get_conversation_history(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT conversation_id, timestamp, role, content, tool_call_id, tool_name, extracted_sql, reasoning"
            " FROM message_history ORDER BY timestamp"
        )
        rows = cursor.fetchall()

        convos = {}
        for conv_id, ts, role, content, tool_call_id, tool_name, extracted_sql, reasoning in rows:
            if conv_id not in convos:
                convos[conv_id] = {
                    'id': conv_id,
                    'createdAt': ts,
                    'name': f"conversation {conv_id}",
                    'messages': []
                }
            msg = {'role': role, 'text': content, 'timestamp': ts}
            # Attach SQL and reasoning if present
            if role == 'assistant':
                if extracted_sql:
                    msg['sql'] = extracted_sql
                if reasoning:
                    msg['reasoning'] = reasoning
            convos[conv_id]['messages'].append(msg)

        cursor.close()
        return list(convos.values())

    def get_conversation_by_id(self, user_id, conversation_id):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT user_id FROM message_history WHERE conversation_id = ?",
            (str(conversation_id),)
        )
        row = cursor.fetchone()
        if not row or row[0] is None:
            raise ValueError(f"Conversation '{conversation_id}' not found.")

        cursor.execute(
            "SELECT timestamp, role, content, tool_call_id, tool_name, extracted_sql, reasoning"
            " FROM message_history WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        rows = cursor.fetchall()

        user_question = None
        assistant_sql = None
        assistant_reasoning = None
        assistant_content = None
        query_id = None
        print(rows[-1])
        for ts, role, content, tool_call_id, tool_name, extracted_sql, reasoning in rows:
            print(f"////////////////{role}\n", content)
            if role == 'user' and user_question is None:
                user_question = content
            elif role == 'assistant':
                # Prefer stored extracted_sql
                assistant_content = content
                if extracted_sql:
                    assistant_sql = extracted_sql
                    assistant_reasoning = reasoning
                    query_id = tool_call_id
                # Fallback: detect SQL in content by simple pattern
                if assistant_sql is None and content and content.strip().endswith(';'):
                    assistant_sql = content.strip()

        response_item = {
            "id": conversation_id,
            "threadId": conversation_id,
            "question": user_question,
            "sql": assistant_sql,
            "view": None,
            "breakdownDetail": None,
            "answerDetail": {
                "queryId": query_id,
                "status": "FINISHED",
                "content": assistant_content,
                "reasoning": assistant_reasoning,
                "numRowsUsedInLLM": 1,
                "error": None,
                "__typename": "ThreadResponseAnswerDetail"
            },
            "chartDetail": None,
            "askingTask": None,
            "adjustment": None,
            "adjustmentTask": None,
            "__typename": "ThreadResponse"
        }

        thread_data = {
            "id": conversation_id,
            "responses": [response_item] if user_question else [],
            "__typename": "DetailedThread"
        }
        cursor.close()
        return {"data": {"thread": thread_data}}

    def get_table_schemas(self, ):
        """
         Returns JSON-compatible dict:
         {"tables": [
             {"table": "tab10", "columns": [{"name":"col1","type":"VARCHAR"},...],
              "sample_rows": [ {col: val,...}, ... up to 5 rows ]
             },
             ...
         ]}
         """
        # fetch schema info
        ddl = self.con.sql(
            "SELECT table_name, column_name, data_type"
            " FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema='public'"
        ).df()

        tables = []
        for table, group in ddl.groupby("table_name"):
            # columns
            cols = [{"name": row["column_name"], "type": row["data_type"]} for _, row in group.iterrows()]
            # sample rows
            sample_df = self.con.sql(f"SELECT * FROM {table} LIMIT 5").df()
            sample = sample_df.to_dict(orient="records")
            tables.append({"table": table, "columns": cols, "sample_rows": sample})

        return {"tables": tables}

    def get_user_clarification(self, clarification_questions: str) -> dict:
        """
        When invoked, returns a JSON dict containing the clarification question
        that should be posed back to the user.
        """
        return {"clarification": clarification_questions}

    def get_few_shot_and_docs(self, question: str, kwargs: dict = None) -> dict:
        """
        Query the RAG store: return similar question-SQL pairs and related docs.
        """
        question_sql_list = self.get_similar_question_sql(question, **(kwargs or {}))
        doc_list = self.get_related_documentation(question, **(kwargs or {}))
        return {"examples": question_sql_list, "docs": doc_list}

    def execute_sql(self, sql):
        extracted_sql = self.extract_sql(sql)

        if not self.is_sql_valid(extracted_sql):
            return "your SQL is not valid"

        try:
            result = self.con.sql(query=sql)
            df = result.df()
            return df
        except Exception as e:
            msg = f"got an exception while executing the sql: {e}"
            print(msg)
            return msg

    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """
        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None

        dfs_uploaded = kwargs.pop('dfs', None)
        user_id = kwargs.get("user_id", "no_user")
        conversation_id = kwargs.get("conversation_id", "no id")
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            dfs=dfs_uploaded,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        for message in prompt:
            self.insert_message_db(conversation_id, user_id, role=message["role"], content=message["content"])
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)
                    # TODO: add logic to handle uploaded spreadsheet
                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list + [
                            f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"

        extracted_sql = self.extract_sql(llm_response)

        return self.fix_sql_case(extracted_sql)

    def extract_sql(self, llm_response: str) -> str:
        """

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
        sqls = re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
        sqls = re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return ""

    def fix_sql_case(self, sql: str) -> str:
        """
        Update SQL conditions:
          - Wrap numeric literals for 'state' with quotes.
          - Convert values for 'table_name' and 'err' checks to uppercase.
        """
        # Fix state: if state is compared to a number, wrap it in quotes.
        # This regex finds "state = <number>" (ignoring case) and replaces it with "state = '<number>'"
        sql = re.sub(
            r"(\bstate\s*=\s*)(\d+)(\b)",
            lambda m: f"{m.group(1)}'{m.group(2)}'",
            sql,
            flags=re.IGNORECASE
        )

        # Fix table_name: ensure that the value is uppercase.
        # This regex finds "table_name = 'value'" (ignoring case) and replaces it with "table_name = 'VALUE'"
        sql = re.sub(
            r"(\btable_name\s*=\s*')([^']+)(')",
            lambda m: f"{m.group(1)}{m.group(2).upper()}{m.group(3)}",
            sql,
            flags=re.IGNORECASE
        )

        # Fix err: if present, ensure that the value is uppercase.
        sql = re.sub(
            r"(\berr\s*=\s*')([^']+)(')",
            lambda m: f"{m.group(1)}{m.group(2).upper()}{m.group(3)}",
            sql,
            flags=re.IGNORECASE
        )

        return sql

    def is_sql_valid(self, sql: str) -> bool:
        """

        Checks if the SQL query is valid. This is usually used to check if we should run the SQL query or not.
        By default, it checks if the SQL query is a SELECT statement.

        Args:
            sql (str): The SQL query to check.

        Returns:
            bool: True if the SQL query is valid, False otherwise.
        """

        parsed = sqlparse.parse(sql)

        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True

        return False

    def should_generate_chart(self, df: pd.DataFrame) -> bool:
        """
        Determines whether a chart should be generated for the given DataFrame based on its content.

        The logic considers:
        1. A single cell (1x1) is allowed (e.g., for a card or gauge).
        2. If the DataFrame is empty, no chart is generated.
        3. For a DataFrame with a single column:
           - If multiple rows exist, the column must have more than one unique value to be interesting
             (this applies to numeric, datetime, or categorical data).
        4. For a DataFrame with multiple columns:
           - If there is exactly one row, it is plottable (e.g., as a bar chart comparing the columns).
           - With multiple rows, at least one column (numeric, datetime, or categorical) must vary.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Returns:
            bool: True if the DataFrame has sufficient variability and structure to be plotted, False otherwise.
        """
        # Empty DataFrame cannot be plotted.
        if df.empty:
            return False

        # Single cell (1x1) is plottable (e.g., as a card or gauge).
        if df.shape == (1, 1):
            return True

        # Handle DataFrames with a single column.
        if df.shape[1] == 1:
            col = df.columns[0]
            # A single row (1x1) case is already handled above.
            if df.shape[0] == 1:
                return True
            # For multiple rows, check if the single column has variation.
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                return df[col].nunique(dropna=True) > 1
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                return df[col].nunique(dropna=True) > 1
            # For any other dtypes, assume not plottable.
            return False

        # For DataFrames with multiple columns, if there is exactly one row,
        # the data can be pivoted (e.g., to a bar chart of metrics) regardless of per-column variation.
        if df.shape[0] == 1:
            return True

        # For DataFrames with multiple rows and columns, check for variation in:
        # Numeric, datetime, or categorical columns.
        numeric_valid = any(
            df[col].nunique(dropna=True) > 1 for col in df.select_dtypes(include='number').columns
        )
        datetime_valid = any(
            df[col].nunique(dropna=True) > 1 for col in df.select_dtypes(include=['datetime', 'datetimetz']).columns
        )
        categorical_valid = any(
            df[col].nunique(dropna=True) > 1 for col in df.select_dtypes(include=['object', 'category']).columns
        )

        return numeric_valid or datetime_valid or categorical_valid

    def generate_rewritten_question(self, last_question: str, new_question: str, **kwargs) -> str:
        """

        Generate a rewritten question by combining the last question and the new question if they are related. If the new question is self-contained and not related to the last question, return the new question.

        Args:
            last_question (str): The previous question that was asked.
            new_question (str): The new question to be combined with the last question.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The combined question if related, otherwise the new question.
        """
        if last_question is None:
            return new_question

        prompt = [
            self.system_message(
                "Your goal is to combine a sequence of questions into a singular question if they are related. If the second question does not relate to the first question and is fully self-contained, return the second question. Return just the new combined question with no additional explanations. The question should theoretically be answerable with a single SQL statement."),
            self.user_message("First question: " + last_question + "\nSecond question: " + new_question),
        ]

        return self.submit_prompt(prompt=prompt, **kwargs)

    def generate_followup_questions(
            self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5, **kwargs
    ) -> list:
        """
        Generate a list of followup questions that you can ask Copilot.

        Args:
            question (str): The question that was asked.
            sql (str): The LLM-generated SQL query.
            df (pd.DataFrame): The results of the SQL query.
            n_questions (int): Number of follow-up questions to generate.

        Returns:
            list: A list of followup questions that you can ask Copilot.
        """

        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe SQL query for this question was: {sql}\n\nThe following is a pandas DataFrame with the results of the query: \n{df.head(25).to_markdown()}\n\n"
            ),
            self.user_message(
                f"Generate a list of {n_questions} followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions. Remember that there should be an unambiguous SQL query that can be generated from the question. Prefer questions that are answerable outside of the context of this conversation. Prefer questions that are slight modifications of the SQL query that was generated that allow digging deeper into the data. Each question will be turned into a button that the user can click to generate a new SQL query so don't use 'example' type questions. Each question must have a one-to-one correspondence with an instantiated SQL query." +
                self._response_language()
            ),
        ]

        llm_response = self.submit_prompt(message_log, **kwargs)

        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    def generate_questions(self, **kwargs) -> List[str]:
        """
        Generate a list of questions that you can ask Copilot.
        """
        question_sql = self.get_similar_question_sql(question="", **kwargs)

        return [q["question"] for q in question_sql]

    def _sanitize_html(self, text: str, token_limit: int = 5) -> str:
        """
        replaces any multiple consecutive '\n' with a single <br> tag for better readability
        :param text:str: txt to sanitize
        :param token_limit: number of tokens to consider when there is separated consecutive '\n'
        :return: the sanitized string
        """
        if '\n' not in text:
            return text

        pattern = r'\n(?:\s*[^\n]{0,' + str(token_limit - 1) + r'}\s*\n)+'
        text = re.sub(pattern, '<br>', text)
        text = text.replace('\n', '')

        return text

    def generate_summary(self, question: str, sql: str, df: pd.DataFrame, **kwargs) -> str:
        """
        Generate a summary of the results of a SQL query.

        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.

        Returns:
            str: The summary of the results of the SQL query.
        """

        message_log = [
            self.system_message(
                f"You are a helpful data assistant for an industrial company that specialises in automotive metal manufacturing.\n\n"
            ),
            self.system_message(
                f"here is the related documentation used in the company: {self.get_related_documentation}\n\n"
            ),
            self.system_message(
                f"The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "give a brief insightful report on the data generated based on the question asked. \n\ndont mention anything about the sql query just the data." +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)
        summary_html = markdown.markdown(summary)
        summary_sanitized = self._sanitize_html(summary_html, 7)
        self.log(summary_sanitized, "Generated Summary:")
        return summary_sanitized

    # ----------------- Use Any Embeddings API ----------------- #
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        pass

    # ----------------- Use Any Database to Store and Retrieve Context ----------------- #
    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """

        This method is used to get all the training data from the retrieval layer.

        Returns:
            pd.DataFrame: The training data.
        """
        pass

    @abstractmethod
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """

        This method is used to remove training data from the retrieval layer.

        Args:
            id (str): The ID of the training data to remove.

        Returns:
            bool: True if the training data was removed, False otherwise.
        """
        pass

    # ----------------- Use Any Language Model API ----------------- #

    @abstractmethod
    def system_message(self, message: str) -> any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        pass

    @abstractmethod
    def assistant_message(self, message: str, tool_call: dict) -> any:
        pass

    @abstractmethod
    def tool_message(self, name, call_id, tool_out) -> any:
        pass

    @abstractmethod
    def tool_call(self, call_id, call_type, function_name, function_args):
        pass

    @abstractmethod
    def use_agentic_mode(self, user_question: str, user_id: str, conversation_id: str = None):
        pass

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

    def add_ddl_to_prompt(
            self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"

            for ddl in ddl_list:
                if (
                        self.str_to_approx_token_count(initial_prompt)
                        + self.str_to_approx_token_count(ddl)
                        < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    def add_documentation_to_prompt(
            self,
            initial_prompt: str,
            documentation_list: list[str],
            max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"

            for documentation in documentation_list:
                if (
                        self.str_to_approx_token_count(initial_prompt)
                        + self.str_to_approx_token_count(documentation)
                        < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
            self, initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"

            for question in sql_list:
                if (
                        self.str_to_approx_token_count(initial_prompt)
                        + self.str_to_approx_token_count(question["sql"])
                        < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt

    def get_sql_prompt(
            self,
            initial_prompt: str,
            question: str,
            question_sql_list: list,
            ddl_list: list = None,
            doc_list: list = None,
            **kwargs,
    ):
        """

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        mytables = kwargs.get('dfs', None)

        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
                             "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        if ddl_list != None:
            initial_prompt = self.add_ddl_to_prompt(
                initial_prompt, ddl_list, max_tokens=self.max_tokens
            )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        if doc_list != None:
            initial_prompt = self.add_documentation_to_prompt(
                initial_prompt, doc_list, max_tokens=self.max_tokens
            )

        if not len(mytables) == 0:
            for i, table in enumerate(mytables):
                table_name = f'{table=}'.split('=')[0]
                table_name += f"{i}"
                initial_prompt += "The following columns are in the table named '" + table_name + "' \n"

                buf = io.StringIO()
                table.info(buf=buf)
                df_info_str = buf.getvalue()
                table_columns = df_info_str.split("Data columns")[1]
                initial_prompt += "Data_columns: " + table_columns + "\n```"

                initial_prompt += "Here is a sample of data that is in the table:\n"
                initial_prompt += "```\n" + table.head(5).to_markdown() + '```\n'
                initial_prompt += f'make sure to use {table_name} if its possible.\n'

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {self.dialect}-compliant and executable, and free of syntax errors. \n"
            "7. Make sure you are using the correct table(s) name(s) in the query. \n"
            "8. If the question is a count make sure to give the count column a meaningful name depending on the question \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    def get_followup_questions_prompt(
            self,
            question: str,
            question_sql_list: list,
            ddl_list: list,
            doc_list: list,
            **kwargs,
    ) -> list:
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions."
            )
        )

        return message_log

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """

        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    def _extract_python_code(self, markdown_string: str) -> str:
        # Strip whitespace to avoid indentation errors in LLM-generated code
        markdown_string = markdown_string.strip()

        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def generate_plotly_code(
            self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:

        df_sample = kwargs.get("df_sample", pd.DataFrame)

        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}\n"
        if not df_sample.empty:
            system_msg += f"The following is a sample of the data in the pandas DataFrame 'df': \n{df_sample}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Choose the most meaningful chart that can be generated with the provided dataframe. Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)
        self.log(title="Generated Plotly Code", message=plotly_code)

        final_code = self._sanitize_plotly_code(self._extract_python_code(plotly_code))
        return final_code

    def generate_common_plotly(self, df: pd.DataFrame) -> str:

        code = ""
        if df.shape == (1, 1):
            code = """import plotly.graph_objects as go

column_name = df.columns[0]  # Get the column name
unique_values = df[column_name].dropna().unique()  # Get unique non-null values

if len(unique_values) == 1:
    value = unique_values[0]  # Extract the unique value

    # Create the indicator plot
    fig = go.Figure(go.Indicator(
        mode="number",
        value=value,
        number={'font': {'size': 80}},  # Adjust font size
        title={'text': column_name, 'font': {'size': 30}},  # Set column name as title
    ))

    fig.show()"""

        final_code = self._sanitize_plotly_code(self._extract_python_code(code))
        return final_code

    def connect_to_database(
            self,
            host: str = None,
            dbname: str = None,
            user: str = None,
            password: str = None,
            port: int = None,
            **kwargs
    ):
        """
        Connect to PostgreSQL using DuckDB's external table functionality.
        This method creates a DuckDB in-memory connection and attaches a PostgreSQL
        database so that queries can be run over both CSV-loaded DuckDB tables and
        PostgreSQL tables seamlessly.

        Args:
            host (str): The PostgreSQL host.
            dbname (str): The PostgreSQL database name.
            user (str): The PostgreSQL user.
            password (str): The PostgreSQL password.
            port (int): The PostgreSQL port.
        """

        # Fetch parameters from environment if not provided
        if not host:
            host = os.getenv("HOST")
        if not host:
            raise ImproperlyConfigured("Please set your PostgreSQL host")

        if not dbname:
            dbname = os.getenv("DATABASE")
        if not dbname:
            raise ImproperlyConfigured("Please set your PostgreSQL database")

        if not user:
            user = os.getenv("PG_USER")
        if not user:
            raise ImproperlyConfigured("Please set your PostgreSQL user")

        if not password:
            password = os.getenv("PASSWORD")
        if not password:
            raise ImproperlyConfigured("Please set your PostgreSQL password")

        if not port:
            port = os.getenv("PORT")
        if not port:
            raise ImproperlyConfigured("Please set your PostgreSQL port")

        try:
            self.con = duckdb.connect(":memory:")
        except Exception as e:
            raise ValidationError(f"Failed to create DuckDB connection: {e}")

        attach_str = f"hostaddr={host} dbname={dbname} user={user} password={password} port={port}"
        try:
            self.con.sql(f"ATTACH '{attach_str}' AS pg (TYPE postgres, SCHEMA 'public');")
        except Exception as e:
            raise ValidationError(f"Failed to attach PostgreSQL database: {e}")

        try:
            table_list = self.con.sql("""select * from information_schema.tables""")["table_name"].fetchall()
            for table in table_list:
                self.con.sql(f"CREATE OR REPLACE VIEW {table[0]} AS SELECT * FROM pg.public.{table[0]}")
        except Exception as e:
            raise ValidationError("couldn't create views for postgresql in duckdb")

        def run_sql_duckdb(sql: str) -> pd.DataFrame:
            try:
                # DuckDB's .df() method converts the query result to a pandas DataFrame.
                return self.con.sql(sql).df()
            except Exception as e:
                raise ValidationError(f"SQL execution error: {e}")

        # Set properties on self for later usage
        self.dialect = "DuckDB"
        self.run_sql_is_set = True
        self.run_sql = run_sql_duckdb

    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """

        Run a SQL query on the connected database.

        Args:
            sql (str): The SQL query to run.

        Returns:
            pd.DataFrame: The results of the SQL query.
        """

        raise Exception(
            "You need to connect to a database first by running CopilotBase.connect_to_database()"
        )

    def ask(
            self,
            question: Union[str, None] = None,
            print_results: bool = True,
            auto_train: bool = True,
            visualize: bool = True,  # if False, will not generate plotly code
            allow_llm_to_see_data: bool = False,
    ) -> Union[
        Tuple[
            Union[str, None],
            Union[pd.DataFrame, None],
            Union[plotly.graph_objs.Figure, None],
        ],
        None,
    ]:
        """

        Ask Copilot a question and get the SQL query that answers it.

        Args:
            question (str): The question to ask.
            print_results (bool): Whether to print the results of the SQL query.
            auto_train (bool): Whether to automatically train Copilot on the question and SQL query.
            visualize (bool): Whether to generate plotly code and display the plotly figure.

        Returns:
            Tuple[str, pd.DataFrame, plotly.graph_objs.Figure]: The SQL query, the results of the SQL query, and the plotly figure.
        """

        if question is None:
            question = input("Enter a question: ")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
        except Exception as e:
            print(e)
            return None, None, None

        if print_results:
            try:
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display(Code(sql))
            except Exception as e:
                print(sql)

        if self.run_sql_is_set is False:
            print(
                "If you want to run the SQL query, connect to a database first."
            )

            if print_results:
                return None
            else:
                return sql, None, None

        try:
            df = self.run_sql(sql)

            if print_results:
                try:
                    display = __import__(
                        "IPython.display", fromList=["display"]
                    ).display
                    display(df)
                except Exception as e:
                    print(df)

            if len(df) > 0 and auto_train:
                self.add_question_sql(question=question, sql=sql)
            # Only generate plotly code if visualize is True
            if visualize:
                try:
                    plotly_code = self.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                    )
                    print("ask ploty code:", plotly_code)
                    fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                    if print_results:
                        try:
                            display = __import__(
                                "IPython.display", fromlist=["display"]
                            ).display
                            Image = __import__(
                                "IPython.display", fromlist=["Image"]
                            ).Image
                            img_bytes = fig.to_image(format="png", scale=2)
                            display(Image(img_bytes))
                        except Exception as e:
                            fig.show()
                except Exception as e:
                    # Print stack trace
                    traceback.print_exc()
                    print("Couldn't run plotly code: ", e)
                    if print_results:
                        return None
                    else:
                        return sql, df, None
            else:
                return sql, df, None

        except Exception as e:
            print("Couldn't run sql: ", e)
            if print_results:
                return None
            else:
                return sql, None, None
        return sql, df, fig

    def train(
            self,
            question: str = None,
            sql: str = None,
            ddl: str = None,
            documentation: str = None,
            plan: TrainingPlan = None,
    ) -> str:
        """
        Train Copilot on a question and its corresponding SQL query.
        If you call it with no arguments, it will check if you connected to a database and it will attempt to train on the metadata of that database.
        If you call it with the sql argument, it's equivalent to [`CopilotBase.add_question_sql()`]
        If you call it with the ddl argument, it's equivalent to [`CopilotBase.add_ddl()`].
        If you call it with the documentation argument, it's equivalent to [`CopilotBase.add_documentation()`].
        Additionally, you can pass a [`TrainingPlan`][CopilotBase.training_plan.TrainingPlan] object. Get a training plan with [`CopilotBase.get_training_plan_generic()`].

        Args:
            question (str): The question to train on.
            sql (str): The SQL query to train on.
            ddl (str):  The DDL statement.
            documentation (str): The documentation to train on.
            plan (TrainingPlan): The training plan to train on.
        """

        if question and not sql:
            raise ValidationError("Please also provide a SQL query")

        if documentation:
            print("Adding documentation....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                print("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def _get_databases(self) -> List[str]:
        try:
            print("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            print(e)
            try:
                print("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                print(e)
                return []

        return df_databases["DATABASE_NAME"].unique().tolist()

    def _get_information_schema_tables(self, database: str) -> pd.DataFrame:
        df_tables = self.run_sql(f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES")

        return df_tables

    def get_training_plan_generic(self, df) -> TrainingPlan:
        """
        This method is used to generate a training plan from an information schema dataframe.

        Basically what it does is breaks up INFORMATION_SCHEMA.COLUMNS into groups of table/column descriptions that can be used to pass to the LLM.

        Args:
            df (pd.DataFrame): The dataframe to generate the training plan from.

        Returns:
            TrainingPlan: The training plan.
        """
        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
            ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column,
                   schema_column,
                   table_column]
        candidates = ["column_name",
                      "data_type",
                      "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                    df.query(f'{database_column} == "{database}"')[schema_column]
                            .unique()
                            .tolist()
            ):
                for table in (
                        df.query(
                            f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                        )[table_column]
                                .unique()
                                .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the table named '{table} ':\n\n"
                    doc += df_columns_filtered_to_table[columns].to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan

    def get_plotly_figure(
            self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        Get a Plotly figure from a dataframe and Plotly code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            plotly_code (str): The Plotly code to use.

        Returns:
            plotly.graph_objs.Figure: The Plotly figure.
        """
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
        except Exception as e:
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                # Use the first two numeric columns for a scatter plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                # Use a bar plot if there's one numeric and one categorical column
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                # Use a pie chart for categorical data with fewer unique values
                fig = px.pie(df, names=categorical_cols[0])
            else:
                # Default to a simple line plot if above conditions are not met
                fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig
