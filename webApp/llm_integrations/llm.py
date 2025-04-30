import os
import pprint
import ast
import sqlite3

from openai import OpenAI

from ..base.base import CopilotBase
from .config import *


class OpenAI_Chat(CopilotBase):
    def __init__(self, client=None, config=None):
        CopilotBase.__init__(self, config=config)

        # default parameters - can be overrided using config
        self.temperature = 0.3

        api_key = "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"
        self.groq_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1/")
        # self._initialize_database()

        if "temperature" in config:
            self.temperature = config["temperature"]

        if client is not None:
            self.client = client
            return

        if config is None and client is None:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return

        if "api_key" in config:
            self.client = OpenAI(api_key=config["api_key"])

    def system_message(self, message: str = "") -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str = "") -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str = "", function_call: dict = None) -> any:
        if function_call is not None:
            return {"role": "assistant", "function_call": function_call}
        return {"role": "assistant", "content": message}

    def tool_message(self, name, call_id, tool_out) -> any:
        return {
            "role": "tool",
            "name": name,
            "tool_call_id": call_id,
            "content": tool_out
        }

    def tool_definition(self, tool_name: str, tool_description: str, properties: dict, required: list) -> dict:
        """
        Build an OpenAI-style function (tool) definition.

        :param tool_name: name of the tool/function
        :param tool_description: human-friendly description of what the tool does
        :param properties: dict mapping parameter names to their JSON schema definitions
        :param required: list of parameter names that are required
        :return: a dict suitable for inclusion in the `tools` list
        """
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            }
        }

    def submit_prompt(self, prompt, **kwargs) -> str:
        # Validate prompt
        if prompt is None or len(prompt) == 0:
            raise Exception("Prompt is empty or None")

        # Estimate token count (approximate)
        num_tokens = sum(len(message["content"]) / 4 for message in prompt)

        # Determine which model/engine to use
        model_or_engine = None
        param_name = None

        # Priority 1: Check kwargs for model or engine
        if kwargs.get("model"):
            model_or_engine = kwargs["model"]
            param_name = "model"
        elif kwargs.get("engine"):
            model_or_engine = kwargs["engine"]
            param_name = "engine"

        # Priority 2: Check self.config for model or engine
        elif self.config is not None:
            if "engine" in self.config:
                model_or_engine = self.config["engine"]
                param_name = "engine"
            elif "model" in self.config:
                model_or_engine = self.config["model"]
                param_name = "model"

        # Priority 3: Default model selection based on token count
        if model_or_engine is None:
            model_or_engine = "gpt-3.5-turbo-16k" if num_tokens > 3500 else "gpt-3.5-turbo"
            param_name = "model"

        # Log the model/engine being used
        print(f"Using {param_name} {model_or_engine} for {num_tokens} tokens (approx)")

        # Create the completion with the appropriate parameter
        params = {
            param_name: model_or_engine,
            "messages": prompt,
            "stop": None,
            "temperature": self.temperature
        }
        # print("\n============prompt sent============\n")
        # print(params)
        # print("\n===============================\n")
        response = self.client.chat.completions.create(**params)

        # Return the first response with text, or the first response's content
        for choice in response.choices:
            if "text" in choice:
                return choice.text

        return response.choices[0].message.content

    # def _initialize_database(self):
    #     """Initializes the SQLite database connection and creates the table if needed."""
    #     try:
    #         self.conn = sqlite3.connect(self.db_path)
    #         self.cursor = self.conn.cursor()
    #         self.cursor.execute("""
    #             CREATE TABLE IF NOT EXISTS message_history (
    #                 id INTEGER PRIMARY KEY AUTOINCREMENT,
    #                 conversation_id TEXT NOT NULL,
    #                 user_id TEXT NOT NULL,
    #                 timestamp DATETIME NOT NULL,
    #                 role TEXT NOT NULL,
    #                 content TEXT NOT NULL,
    #                 tool_call_id TEXT, -- Optional: For linking tool requests and responses
    #                 tool_name TEXT     -- Optional: Name of the tool called/responded
    #             )
    #         """)
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

    def get_available_tools(self):

        tools = [
            self.tool_definition(
                tool_name="get_table_schemas",
                tool_description="Return a list of tables available to query, each with its columns and data types and a 5 row sample of the data, it should be used when you are ready to build the SQL query and you need the information about the tables to query, dont use it unless the question is clear.",
                properties={},
                required=[]
            ),
            self.tool_definition(
                tool_name="get_user_clarification",
                tool_description="This tool is used to ask the user for further clarification, it should be used when the initial question is ambiguous, it has one parameter 'clarification_question' which you should provide to ask it to the user, example: user: 'what is the trs of today', clarification_question: 'can define what does the TRS stand for?'",
                properties={
                    "clarification_questions": {"type": "string",
                                                "description": "The questions to ask the user so he can provide more clarification"}
                },
                required=["clarification_questions"]
            ),
            self.tool_definition(
                tool_name="get_few_shot_and_docs",
                tool_description="Retrieve similar example SQL queries with their proper question and related documentation for the user's question from a RAG DB, it should be used when you need further context.",
                properties={
                    "question": {"type": "string", "description": "The user's original question"},
                    "kwargs": {"type": "object",
                               "description": "Optional parameters for similarity search and documentation retrieval"}
                },
                required=["question"]
            )
        ]
        return tools

    def use_tools(self, tool_calls):
        call = tool_calls[0]
        name = call.function.name
        args = ast.literal_eval(call.function.arguments) or {}

        print(f"model called {name} tool")
        print(f"the args returned are: {args}")

        if name == "get_table_schemas":
            tool_out = self.get_table_schemas()
        elif name == "get_user_clarification":
            tool_out = self.get_user_clarification(args["clarification_questions"])
        elif name == "get_few_shot_and_docs":
            tool_out = self.get_few_shot_and_docs(args["question"])
        else:
            raise RuntimeError(f"Unknown tool: {name}")

        print("got this as a result", pprint.pprint(tool_out))
        return tool_out

    def use_agentic_mode(self, user_question):
        """
        Main entry point: given a natural-language question, runs an agentic
        workflow that may call our tools, then returns either a clarification
        prompt or the final SQL string.
        """
        # Initial messages

        messages = [self.system_message(message=system_prompt), self.user_message(user_question),
                    self.assistant_message(
                        function_call={"name": "get_few_shot_and_docs", "arguments": str({"question": user_question})}),
                    self.tool_message(name="get_few_shot_and_docs", call_id="fs1",
                                      tool_out=str(self.get_few_shot_and_docs(user_question))),
                    self.assistant_message(function_call={"name": "get_table_schemas", "arguments": str({})}),
                    self.tool_message(name="get_table_schemas", call_id="sch1",
                                      tool_out=str(self.get_table_schemas()))
                    ]

        print(f"using {model_name} model")
        pprint.pprint(messages)
        while True:

            print("asking the model:")

            resp = self.groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=self.get_available_tools(),
                temperature=0.1
            )
            msg = resp.choices[0].message
            if msg.content == '' and msg.tool_calls is None:
                print("empty response")
                continue

            print("msg returned: ")
            pprint.pprint(msg)
            # If the model wants to call a tool
            if msg.tool_calls:
                tool_out = self.use_tools(msg.tool_calls)
                messages.append(self.assistant_message(function_call={"name": name, "arguments": str(args)}))
                messages.append(self.tool_message(name, call.id, str(tool_out)))

                if name == "get_user_clarification":
                    return str(tool_out["clarification"])

            final_response = msg.content or ""
            sql = self.extract_sql(final_response)
            if sql == "":
                # ask again for valid SQL
                messages.append(self.user_message(
                    "I didn’t detect a valid SQL query—please provide only the SQL statement ending with a semicolon."))
                continue

            try:
                self.con.sql(query=sql)
                return sql
            except Exception as e:
                print("not valid sql: ", sql)
                print(f"the model encountered an error {e} and it will try to fix it")
                messages.append(self.user_message(f"i executed the sql query and got this error:```{e}```"))
                continue
