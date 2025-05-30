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
import uuid  # Added for generating conversation IDs

import pandas as pd
import duckdb
import io
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlparse
from flask import g, has_app_context

from .exceptions import ImproperlyConfigured, ValidationError
from .training_plan import TrainingPlan, TrainingPlanItem

CONVERSATION_DB_PATH = r"C:\Users\ACER\PycharmProjects\LogisticRegression\env\text2sql\webApp\data\conversations.db"
thread_local = threading.local()


def get_connection():
    """
    If we are in a Flask request context, reuse g.db_conn (or open it).
    Otherwise fall back to a thread‑local connection.
    """
    if has_app_context():
        # Flask request → store in g
        if not hasattr(g, 'db_conn'):
            conn = sqlite3.connect(CONVERSATION_DB_PATH,
                                   timeout=30.0,
                                   check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            g.db_conn = conn
        return g.db_conn
    else:
        # outside Flask → thread‑local
        conn = getattr(thread_local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(CONVERSATION_DB_PATH,
                                   timeout=30.0,
                                   check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.commit()
            thread_local.conn = conn
        return conn


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

    def create_new_conversation(self, user_question: str, user_id: str, **kwargs) -> str:
        """
        Creates a new conversation, stores the initial system prompt, user question,
        and simulated tool interaction as context, and returns the conversation ID.
        Tool calls are represented as normal assistant/user messages.

        Args:
            user_question (str): The initial question from the user.
            user_id (str): The ID of the user initiating the conversation.

        Returns:
            str: The newly generated conversation ID.
        """
        self.log(f"Creating new conversation (contextual tools) for user_id: {user_id} with question: {user_question}")
        conversation_id = str(uuid.uuid4())

        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None

        dfs_uploaded = kwargs.pop('dfs', None)
        question_sql_list = self.get_similar_question_sql(user_question, **kwargs)
        doc_list = self.get_related_documentation(user_question, **kwargs)
        messages = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=user_question,
            question_sql_list=question_sql_list,
            doc_list=doc_list,
            dfs=dfs_uploaded,
            **kwargs,
        )

        for msg in messages:
            self.insert_message_db(
                conversation_id=conversation_id,
                user_id=user_id,
                role=msg['role'],
                content=msg['content']
            )
        return conversation_id

    # ----database related----

    def insert_message_db(self, conversation_id, user_id, role=None, tool_call_id=None, content=None, tool_calls=None,
                          extracted_sql=None, reasoning=None):
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
        try:
            cursor.execute(
                """INSERT INTO message_history (timestamp,conversation_id,user_id,role,tool_call_id,content,tool_calls,extracted_sql,reasoning)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (timestamp, conversation_id, user_id, role, tool_call_id, content, str(tool_calls), extracted_sql,
                 reasoning)
            )
            conn.commit()
        finally:
            cursor.close()
            # We don't close the connection here as it's managed by Flask's app context

    def get_conversation_history(self, user_id):
        conn = get_connection()
        cursor = conn.cursor()
        all_conversations_data = []

        try:
            # Step 1: Get all distinct conversation IDs
            cursor.execute("SELECT DISTINCT conversation_id FROM message_history ORDER BY timestamp DESC")
            distinct_conv_ids_rows = cursor.fetchall()
            distinct_conv_ids = [row[0] for row in distinct_conv_ids_rows]

            if not distinct_conv_ids:
                return []  # No conversations found

            # Step 2: For each conversation_id, get its messages
            for conv_id in distinct_conv_ids:
                cursor.execute(
                    "SELECT timestamp, role, content, tool_calls, extracted_sql, reasoning "
                    "FROM message_history "
                    "WHERE conversation_id = ? "
                    "ORDER BY timestamp ASC",  # Order messages by timestamp to easily find first and last
                    (conv_id,)
                )
                messages_rows = cursor.fetchall()

                if not messages_rows:
                    # This case should ideally not happen if conv_id comes from the table itself,
                    # but as a safeguard:
                    print(f"Warning: No messages found for conversation_id {conv_id}, skipping.")
                    continue

                conversation_messages = []
                # The first message's timestamp is the conversation's creation time
                conversation_created_at = messages_rows[0][0]  # timestamp is the first column (index 0)

                # Default name is the conversation ID
                conversation_name = str(conv_id)

                for ts, role, content, tool_calls, extracted_sql, reasoning in messages_rows:
                    # Update conversation name if the message is from a user and has content
                    # Since messages are ordered by timestamp ASC, the last user message encountered
                    # will set the final name.
                    if role == 'user' and content:
                        conversation_name = content

                    msg = {'role': role, 'text': content, 'timestamp': ts}

                    # Attach SQL and reasoning if present for assistant messages
                    if role == 'assistant':
                        if extracted_sql:
                            msg['sql'] = extracted_sql
                        if reasoning:
                            msg['reasoning'] = reasoning

                    conversation_messages.append(msg)

                # Construct the conversation object
                convo_obj = {
                    'id': conv_id,
                    'createdAt': conversation_created_at,
                    'name': conversation_name,
                    'messages': conversation_messages
                }
                all_conversations_data.append(convo_obj)

            # all_conversations_data.sort(key=lambda c: c['createdAt'], reverse=True)

            return all_conversations_data

        except Exception as e:
            print(f"An error occurred: {e}")
            # Depending on error handling strategy, you might want to raise the exception,
            # return None, or return an empty list.
            return []  # Or raise e
        finally:
            if cursor:
                cursor.close()

    def get_conversation_by_id(self, user_id, conversation_id):
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT DISTINCT user_id FROM message_history WHERE conversation_id = ?",
                (str(conversation_id),)
            )
            row = cursor.fetchone()

            # Check if conversation exists and user has access
            if not row or (row[0] is None):
                raise ValueError(f"Conversation '{conversation_id}' not found.")

            cursor.execute(
                "SELECT timestamp, role, content, tool_calls, extracted_sql, reasoning"
                " FROM message_history WHERE conversation_id = ? ORDER BY timestamp",
                (conversation_id,)
            )
            rows = cursor.fetchall()

            user_question = None
            assistant_sql = None
            assistant_reasoning = None
            assistant_content = None
            query_id = None
            for ts, role, content, tool_calls, extracted_sql, reasoning in rows:
                if role == 'user' :
                    user_question = content
                    assistant_content = None
                    assistant_sql = None
                    assistant_reasoning = None
                elif role == 'assistant':
                    # Prefer stored extracted_sql
                    assistant_content = content
                    if extracted_sql:
                        assistant_sql = extracted_sql
                        assistant_reasoning = reasoning
                    # Fallback: detect SQL in content by simple pattern
                    if assistant_sql is None and content and content.strip().endswith(';'):
                        assistant_sql = content.strip()
            status = "FINISHED" if assistant_sql and assistant_sql != "" and assistant_content and assistant_content != '' else "STREAMING"
            response_item = {
                "id": conversation_id,
                "threadId": conversation_id,
                "question": user_question,
                "sql": assistant_sql,
                "view": None,
                "breakdownDetail": None,
                "answerDetail": {
                    "queryId": query_id,
                    "status": status,
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
            return {"data": {"thread": thread_data}}
        finally:
            cursor.close()

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
        cursor = self.con.cursor()
        try:
            ddl = cursor.sql(
                "SELECT table_name, column_name, data_type "
                "FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema='public'"
            ).df()

            tables = []
            for table, group in ddl.groupby("table_name"):
                cols = [{"name": r["column_name"], "type": r["data_type"]}
                        for _, r in group.iterrows()]
                sample_df = cursor.sql(f"SELECT * FROM {table} LIMIT 1").df()
                sample = sample_df.to_dict(orient="records")
                tables.append({"table": table, "columns": cols, "sample_rows": sample})
            return {"tables": tables}
        finally:
            cursor.close()

    def create_or_load_conversation(self, conversation_id: str, user_id: str) -> list:
        """
        Load prior messages from DB and convert to LLM format.
        """
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role,tool_call_id, content, tool_calls, extracted_sql, reasoning"
            " FROM message_history WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        msgs = []
        rows = cursor.fetchall()
        tool_name = ""
        for role, tool_call_id, content, tool_calls, extracted_sql, reasoning in rows:
            if role == 'assistant' and tool_calls and tool_calls != "None":
                # reconstruct function call messages if any
                args = ast.literal_eval(tool_calls)[0]
                tool_name = args['function']['name']
                msg = self.assistant_message("", args)
                msgs.append(msg)
            elif role == 'tool':
                msg = self.tool_message(name=tool_name, call_id=tool_call_id, tool_out=content)
                msgs.append(msg)
                tool_name = ""
            else:
                # system/user/assistant normal
                msg = {'role': role, 'content': content}
                msgs.append(msg)
                tool_name = ""

        return msgs

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

    def generate_sql(self, user_id, conversation_id, **kwargs) -> str:
        """
        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        messages = self.create_or_load_conversation(conversation_id=conversation_id, user_id=user_id)
        llm_response = self.submit_prompt(messages, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        extracted_sql = self.extract_sql(llm_response)

        return self.fix_sql_case(extracted_sql)

    def execute_sql(self, sql):
        extracted_sql = self.extract_sql(sql)

        if not self.is_sql_valid(extracted_sql):
            return "your SQL is not valid"

        cursor = self.con.cursor()
        try:
            result = cursor.sql(query=sql)
            df = result.df()
            return df
        except Exception as e:
            msg = (f"got an exception while executing the sql: ```{e}```.",
                   "please fix the error and re-generate a new SQL query and try executing again using 'exectue_sql tool.'"
                   )
            print(msg)
            return msg
        finally:
            cursor.close()

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
                f"here is the related documentation used in the company: {self.get_related_documentation(question)}\n\n"
            ),
            self.system_message(
                f"The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "give a brief insightful report on the data generated based on the question asked. \n\ndont mention anything about the sql query just the data." +
                self._response_language()
            ),
        ]
        for chunk in self.submit_prompt_stream(message_log):
            yield chunk

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
    def tool_call(self, call_id, function_name, function_args):
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
    ) -> list:
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

    def _sanitize_vega_code(self, raw_vega_code: str) -> str:
        """
        - Drops any bare 'chart' or 'chart.show()' at the end of the snippet.
        - Ensures the snippet ends in a chart binding for .to_dict().
        """
        lines = raw_vega_code.strip().splitlines()
        # drop trailing 'chart' or 'chart.show()'
        while lines and lines[-1].strip() in {"chart", "chart.show()"}:
            lines.pop()
        return "\n".join(lines) + "\n"

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

    def generate_vega_code(self, question: str = None, sql: str = None, df: pd.DataFrame = None, error: dict = None):

        system_msg = "You are an expert in Python and the Altair data visualization library.",
        "Your task is to generate Python code that defines an Altair chart object.",
        "The chart will be rendered in Grafana, which will provide the data.",
        "The Python code should assign the Altair chart object to a variable named `chart`.",
        "Do NOT include any data loading (e.g., pandas.read_csv) or `chart.show()` calls.",
        "The chart should be defined to expect data, e.g., `alt.Chart().mark_...` not `alt.Chart(my_dataframe)...`.",
        "Use `alt.X`, `alt.Y`, `alt.Color`, etc., for encoding definitions.",
        "Ensure field names in encodings exactly match the provided column names.",
        "Make the chart interactive using `.interactive()`.",

        if question is not None:
            system_msg = f"The user asked the following question: '{question}'"
        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this Duckdb SQL query: {sql}\n\n"
        if df is not None:
            system_msg += f"\n\nThis is a sample of the data inside the dataframe: {df.head(10)}\n\n"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Generate a Python script to chart the results of the dataframe."
                "you may only use Altair Vega library to generate the charts"
                "Choose the most meaningful chart that can help answering the question using the resulting dataframe."
                "Assume the data is in a pandas dataframe called 'df'."
                "If there is only one value in the dataframe, use an Indicator."
                "Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        if error is not None and 'generated_code' in list(error.keys()) and 'error' in list(error.keys()):
            message_log.append(self.assistant_message(message=error['generated_code']))
            message_log.append(self.user_message(
                message=f"i tried executing your code but i got the following error:\n{error['error']}"))

        vega_code = self.submit_prompt(message_log)
        print("generated code: \n", vega_code)
        self.log(title="Generated vega Code", message=vega_code)
        final_code = self._sanitize_vega_code(self._extract_python_code(vega_code))
        return final_code

    def get_vega_spec(self, question: str = None, sql: str = None, df: pd.DataFrame = None) -> dict:
        """
        Execs the generated Altair code and returns the Vega-Lite spec JSON.
        """
        # locals dict: give exec access to df and alt
        ldict = {"df": df, "alt": __import__("altair")}
        error = {}
        for i in range(3):

            code = self.generate_vega_code(
                question=question,
                sql=sql,
                df=df.head(5),
                error=error
            )
            error['generated_code'] = code

            try:
                exec(code, globals(), ldict)
                chart = ldict.get("chart", None)
                continue
            except Exception as e:
                error['error'] = f'{e}'
                raise RuntimeError("error generating the vega chart: ", e)

        if chart is None:
            raise RuntimeError("No Altair chart produced by vega_code")

        print("the returned chart:\n", chart)
        # Return the Vega-Lite spec as a dict
        return chart

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
        cursor = self.con.cursor()
        try:
            cursor.sql(f"ATTACH '{attach_str}' AS pg (TYPE postgres, SCHEMA 'public');")

        except Exception as e:
            cursor.close()
            raise ValidationError(f"Failed to attach PostgreSQL database: {e}")

        try:
            table_list = cursor.sql("""select * from information_schema.tables""")["table_name"].fetchall()
            for table in table_list:
                cursor.sql(f"CREATE OR REPLACE VIEW {table[0]} AS SELECT * FROM pg.public.{table[0]}")

        except Exception as e:
            cursor.close()
            raise ValidationError("couldn't create views for postgresql in duckdb")

        def run_sql_duckdb(sql: str) -> pd.DataFrame:
            try:
                # DuckDB's .df() method converts the query result to a pandas DataFrame.
                df = cursor.sql(sql).df()
                cursor.close()
                return df
            except Exception as e:
                cursor.close()
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
        Generate a TrainingPlan from a DataFrame with columns:
          - table_name
          - column_name
          - data_type
          - (optional) comment

        Groups only by table_name and emits one plan item per table.
        """

        # 1) sanity check
        if 'table_name' not in df.columns:
            raise ValueError("DataFrame must have a 'table_name' column")

        # 2) pick up whichever of column_name/data_type/comment exist
        detail_cols = [c for c in ('column_name', 'data_type', 'comment') if c in df.columns]

        # 3) these are the columns we’ll include in the markdown
        cols = ['table_name'] + detail_cols

        plan = TrainingPlan([])

        # 4) for each table, build the doc and append a plan item
        for table in df['table_name'].unique():
            df_sub = df[df['table_name'] == table]

            # build the markdown: e.g.
            # The following columns are in the table named 'users':
            #
            # | table_name | column_name | data_type |
            doc = f"The following columns are in the table named '{table}':\n\n"
            doc += df_sub[cols].to_markdown(index=False)

            plan._plan.append(
                TrainingPlanItem(
                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                    item_group=table,  # you can also set "" or a fixed value here
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
