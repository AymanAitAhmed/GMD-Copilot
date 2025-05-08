import os
import pprint
import ast
import sqlite3
import re
import uuid
import json

from openai import OpenAI
from openai import APITimeoutError
from pandas import DataFrame

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

    def assistant_message(self, message: str = "", tool_call: dict = None) -> any:
        if tool_call is not None:
            return {"role": "assistant", "tool_calls": [tool_call]}
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

    def tool_call(self, call_id, function_name, function_args):
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": function_name, "arguments": str(function_args)},
        }
        return tool_call

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
            ),
            self.tool_definition(
                tool_name="execute_sql",
                tool_description=(
                    "Executes the given SQL on a DuckDB database and check if the SQL has any errors, this should be used when you want to answer the user question with an SQL, don't use it unless you have sufficient context and you clearly understand the user needs."
                ),
                properties={
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    }
                },
                required=["sql"]
            ),
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
        elif name == "execute_sql":
            tool_out = self.execute_sql(args['sql'])
        else:
            raise RuntimeError(f"Unknown tool: {name}")

        print("got this as a result", tool_out)
        return tool_out

    def _sanitize_summary(self, text: str) -> str:
        # remove <think>...</think> sections
        return re.sub(r"<think>[\s\S]*?<\/think>", "", text).strip()

    def _generate_conversation_id(self) -> str:
        """
        Create a new unique conversation ID.
        """
        return str(uuid.uuid4())

    def _build_initial_messages(self, user_question: str, user_id: str, conversation_id: str) -> list:
        # Prepare the initial prompt sequence and log to DB
        msgs = [
            self.system_message(message=system_prompt),
            self.user_message(user_question),
            self.assistant_message(tool_call=self.tool_call("fs1", "get_few_shot_and_docs",
                                                            function_args={"question": user_question})),
            self.tool_message(name="get_few_shot_and_docs", call_id="fs1",
                              tool_out=str(self.get_few_shot_and_docs(user_question))),

            self.assistant_message(tool_call=self.tool_call("sch1", "get_table_schemas", function_args={})),
            self.tool_message(name="get_table_schemas", call_id="sch1",
                              tool_out=str(self.get_table_schemas()))
        ]
        for m in msgs:
            role = m['role']
            content = m.get('content') or ''
            self.insert_message_db(conversation_id, user_id, role, content,
                                   tool_call_id=m.get('tool_call_id'),
                                   tool_name=m.get('name'))
        return msgs

    def _steps_summarize(self, sql: str, df: DataFrame) -> str:

        # Ask LLM for step-by-step summary
        summary_prompt = (
            "You are a helpful assistant that helps a non technical user that doesnt know anything about SQL."
            "Summarize step-by-step what you did to generate the SQL query. "
            "Include tables used, columns used, any calculations with LaTeX formulas."
            "Don't mention anything about the SQL as the user wouldn't understand it anyway."
            f"Here is the SQL you generated: ```{sql}```"
            "Produce a final answer using the result. "
            f"Here is the query result as JSON: {df.to_dict(orient='records')}"
        )
        summary_resp = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[self.system_message(summary_prompt)],
            temperature=0.1
        )
        raw_summary = summary_resp.choices[0].message.content
        clean_summary = self._sanitize_summary(raw_summary)
        return clean_summary

    def get_or_create_conversation_id(self, conversation_id: str, user_id: str, user_question: str) -> str:
        """
        Returns a valid conversation ID, creating a new one if necessary.
        Does not build messages or perform any LLM operations - just handles ID logic.
        """
        if conversation_id is None:
            print("Creating new conversation ID")
            # Generate a new ID but don't do anything else yet
            conversation_id = self._generate_conversation_id()
            # Add initial user message to the database
            self.insert_message_db(conversation_id, user_id, "user", user_question)
        else:
            print(f"Using existing conversation ID: {conversation_id}")
            
        return conversation_id
            
    def _create_or_load_conversation(self, conversation_id: str, user_id: str, user_question: str) -> (list, str):
        """
        Load prior messages from DB and convert to LLM format.
        """

        if conversation_id is None:
            print("creating new conversation")
            conversation_id = self._generate_conversation_id()
            messages = self._build_initial_messages(user_question, user_id, conversation_id)
            return messages, conversation_id

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, tool_call_id, tool_name, extracted_sql, reasoning"
            " FROM message_history WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        )
        msgs = []
        for role, content, tool_call_id, tool_name, extracted_sql, reasoning in cursor.fetchall():
            if role == 'assistant' and tool_call_id and tool_name:
                # reconstruct function call messages if any
                msgs.append(self.tool_message(tool_name, tool_call_id, content))
            else:
                # system/user/assistant normal
                msg = {'role': role, 'content': content}
                msgs.append(msg)

        user_msg = self.user_message(user_question)
        self.insert_message_db(conversation_id, user_id, "user", user_question)
        msgs.append(user_msg)
        return msgs, conversation_id

    def use_agentic_mode(self, user_question: str, user_id: str, conversation_id: str = None) -> dict:
        """
        Orchestrates the agentic workflow. If conversation_id is None, starts a new
        conversation; otherwise continues an existing one.
        """

        messages, conversation_id = self._create_or_load_conversation(conversation_id, user_id, user_question)

        while True:
            print("asking the model")

            try:
                resp = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=self.get_available_tools(),
                    temperature=0,
                    tool_choice="auto",

                )
            except APITimeoutError as e:
                print("Timeout! Please check internet connection.")
                return {"error": "Timeout! Please check internet connection."}

            msg = resp.choices[0].message
            print("the model answered:", msg)
            # Handle any tool calls
            if msg.tool_calls:
                print("using tools")
                first_call = msg.tool_calls[0]
                tool_out = self.use_tools(msg.tool_calls)
                # Log assistant function call
                args = first_call.function.arguments
                parsed_args = json.loads(args)
                tool_call = self.tool_call(first_call.id,first_call.function.name,parsed_args)
                messages.append(self.assistant_message(tool_call=tool_call))
                self.insert_message_db(conversation_id, user_id, "assistant",
                                       content=f"tool_call: {first_call.function.name}",
                                       tool_call_id=first_call.id,
                                       tool_name=first_call.function.name)

                if first_call.function.name == "get_user_clarification":
                    return {"success": tool_out.get("clarification")}

                if first_call.function.name == "execute_sql" and isinstance(tool_out, DataFrame):
                    print("we got a valid result from DB")
                    sql = ast.literal_eval(args)['sql']
                    summary = self._steps_summarize(sql, tool_out)
                    self.insert_message_db(conversation_id, user_id, "assistant", content=summary,
                                          extracted_sql=sql, reasoning=summary)
                    # Store tables used for API integration
                    tables_used = self._extract_tables_from_sql(sql)
                    return {
                        "success": tool_out.to_markdown(),
                        "sql": sql,
                        "reasoning": summary,
                        "tables": tables_used
                    }
                else:
                    print("invalid sql")
                    # Log tool output
                    messages.append(self.tool_message(first_call.function.name,
                                                      first_call.id,
                                                      str(tool_out)))
                    self.insert_message_db(conversation_id, user_id, "tool",
                                           content=str(tool_out),
                                           tool_call_id=first_call.id,
                                           tool_name=first_call.function.name)

                continue

            if msg.content:
                print("got normal message:", msg)
                messages.append(self.assistant_message(message=msg.content))
                self.insert_message_db(conversation_id, user_id, "assistant", content=msg.content)
                return {"success": msg.content}
            else:
                return {"error": "there was an error communicating with the AI! please try again."}
                
    def _extract_tables_from_sql(self, sql: str) -> list:
        """
        Extract table names from SQL for metadata in the API response.
        """
        # Simple regex extraction of tables after FROM and JOIN keywords
        # This is a basic implementation and may need refinement
        tables = []
        try:
            # Find tables after FROM clause
            from_matches = re.findall(r'\bFROM\s+([\w\._]+)', sql, re.IGNORECASE)
            for match in from_matches:
                table_name = match.strip('"').strip('\'').strip('`')
                if table_name and table_name not in tables:
                    tables.append({"name": table_name, "description": f"Table containing {table_name} data"})
            
            # Find tables after JOIN clause
            join_matches = re.findall(r'\bJOIN\s+([\w\._]+)', sql, re.IGNORECASE)
            for match in join_matches:
                table_name = match.strip('"').strip('\'').strip('`')
                if table_name and table_name not in tables:
                    tables.append({"name": table_name, "description": f"Table containing {table_name} data"})
        except Exception as e:
            print(f"Error extracting tables from SQL: {e}")
        
        return tables
        
    def stream_response(self, user_question: str, user_id: str, conversation_id: str = None):
        """
        Generator function that yields streaming responses from the agentic workflow.
        Designed to work with Server-Sent Events (SSE).
        """
        # Initial status update
        yield json.dumps({"message": "Processing your question...", "done": False})
        
        try:
            messages, conversation_id = self._create_or_load_conversation(conversation_id, user_id, user_question)
            
            # First yield before processing to indicate understanding phase
            yield json.dumps({"message": "Analyzing available data sources...", "done": False})
            
            while True:
                try:
                    resp = self.groq_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        tools=self.get_available_tools(),
                        temperature=0,
                        tool_choice="auto",
                    )
                except APITimeoutError as e:
                    yield json.dumps({"message": "Timeout! Please check internet connection.", "done": True})
                    return
                
                msg = resp.choices[0].message
                
                # Handle tool calls
                if msg.tool_calls:
                    first_call = msg.tool_calls[0]
                    function_name = first_call.function.name
                    
                    # Update status with current tool being used
                    yield json.dumps({"message": f"Using {function_name} to process your question...", "done": False})
                    
                    tool_out = self.use_tools(msg.tool_calls)
                    args = first_call.function.arguments
                    parsed_args = json.loads(args)
                    tool_call = self.tool_call(first_call.id, function_name, parsed_args)
                    
                    messages.append(self.assistant_message(tool_call=tool_call))
                    self.insert_message_db(conversation_id, user_id, "assistant",
                                        content=f"tool_call: {function_name}",
                                        tool_call_id=first_call.id,
                                        tool_name=function_name)
                    
                    # Handle user clarification request
                    if function_name == "get_user_clarification":
                        yield json.dumps({"message": tool_out.get("clarification"), "done": True})
                        return
                    
                    # Handle SQL execution result
                    if function_name == "execute_sql" and isinstance(tool_out, DataFrame):
                        sql = ast.literal_eval(args)['sql']
                        summary = self._steps_summarize(sql, tool_out)
                        self.insert_message_db(conversation_id, user_id, "assistant", content=summary,
                                            extracted_sql=sql, reasoning=summary)
                        
                        # Final response with markdown table
                        yield json.dumps({"message": tool_out.to_markdown(), "done": True})
                        return
                    else:
                        # Continue with intermediate tool output
                        messages.append(self.tool_message(function_name, first_call.id, str(tool_out)))
                        self.insert_message_db(conversation_id, user_id, "tool",
                                            content=str(tool_out),
                                            tool_call_id=first_call.id,
                                            tool_name=function_name)
                        continue
                
                if msg.content:
                    messages.append(self.assistant_message(message=msg.content))
                    self.insert_message_db(conversation_id, user_id, "assistant", content=msg.content)
                    yield json.dumps({"message": msg.content, "done": True})
                    return
                else:
                    yield json.dumps({"message": "There was an error communicating with the AI. Please try again.", "done": True})
                    return
                    
        except Exception as e:
            print(f"Error in streaming response: {str(e)}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"message": f"Error: {str(e)}", "done": True})
