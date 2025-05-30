import os
import pprint
import ast
import sqlite3
import re
import uuid
import json

from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError
from openai.types.chat import ChatCompletionMessage
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
        return {"role": "assistant", "content": message, "tool_calls": [tool_call]}

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
            "function": {"name": function_name, "arguments": f"{function_args}"},
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
        print("\n============prompt sent============\n")
        print(params)
        print("\n===============================\n")
        response = self.client.chat.completions.create(**params)
        print("the response: ", response)
        # Return the first response with text, or the first response's content
        for choice in response.choices:
            if "text" in choice:
                return choice.text

        return response.choices[0].message.content

    def submit_prompt_stream(self, prompt, **kwargs):
        # Validate prompt
        if prompt is None or len(prompt) == 0:
            raise ValueError("Prompt is empty or None")

        # Estimate token count (approximate)
        num_tokens = 0
        if isinstance(prompt, list):
            for message in prompt:
                if isinstance(message, dict) and "content" in message and isinstance(message["content"], str):
                    num_tokens += len(message["content"]) / 4
        elif isinstance(prompt, str):
            num_tokens += len(prompt) / 4

        model_or_engine = None
        param_name = None

        if kwargs.get("model"):
            model_or_engine = kwargs["model"]
            param_name = "model"
        elif kwargs.get("engine"):  # Some models on OpenRouter might still use 'engine'
            model_or_engine = kwargs["engine"]
            param_name = "engine"
        elif self.config is not None:
            if self.config.get("engine"):
                model_or_engine = self.config["engine"]
                param_name = "engine"
            elif self.config.get("model"):
                model_or_engine = self.config["model"]
                param_name = "model"

        if model_or_engine is None:
            # Default to a model known to work well with OpenRouter and streaming
            # For OpenRouter, it's generally 'model'.
            # Using a general purpose model like 'openai/gpt-3.5-turbo'
            model_or_engine = "openai/gpt-3.5-turbo-16k" if num_tokens > 3500 else "openai/gpt-3.5-turbo"
            param_name = "model"

        print(f"Streaming using {param_name} {model_or_engine} for {num_tokens} tokens (approx)")

        params = {
            param_name: model_or_engine,
            "messages": prompt,
            "temperature": self.temperature,
            "stream": True,
            "stop": kwargs.get("stop"),
            "max_tokens": kwargs.get("max_tokens")
        }
        params = {k: v for k, v in params.items() if v is not None}

        print("\n============prompt sent (stream)============\n")
        print(params)
        print("\n===============================\n")

        try:
            stream = self.client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                    received_chunk = chunk.choices[0].delta.content
                    yield received_chunk
        except Exception as e:
            print(f"Error during streaming: {e}")
            # You might want to re-raise the exception or handle it differently
            raise

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

        if name == "get_table_schemas":
            tool_out = self.get_table_schemas()
        elif name == "get_user_clarification":
            tool_out = self.get_user_clarification(args["clarification_questions"])
        elif name == "get_few_shot_and_docs":
            tool_out = self.get_few_shot_and_docs(args["question"])
        elif name == "execute_sql":
            print("trying to use execute_sql")
            tool_out = self.execute_sql(args['sql'])
        else:
            raise RuntimeError(f"Unknown tool: {name}")

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
            self.assistant_message(tool_call=self.tool_call("sch1", "get_table_schemas", function_args={})),
            self.tool_message(name="get_table_schemas", call_id="sch1",
                              tool_out=str(self.get_table_schemas())),
            self.assistant_message(tool_call=self.tool_call("fs1", "get_few_shot_and_docs",
                                                            function_args={"question": user_question})),
            self.tool_message(name="get_few_shot_and_docs", call_id="fs1",
                              tool_out=str(self.get_few_shot_and_docs(user_question))),
        ]
        for m in msgs:
            role = m['role']
            content = m.get('content') or ''
            tool_calls = m.get('tool_calls')
            tool_call_id = m.get('tool_call_id')
            self.insert_message_db(conversation_id, user_id, role, tool_call_id, content,
                                   tool_calls=tool_calls)
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
            # Generate a new ID but don't do anything else yet
            conversation_id = self._generate_conversation_id()
            print("Creating new conversation ID: ", conversation_id)

            # Add initial user message to the database
            # self.insert_message_db(conversation_id, user_id, "user", user_question)
        else:
            print(f"Using existing conversation ID: {conversation_id}")

        return conversation_id

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

    def use_agentic_mode(self, user_question: str, user_id: str, conversation_id: str = None) -> dict:
        """
        Orchestrates the agentic workflow. If conversation_id is None, starts a new
        conversation; otherwise continues an existing one.
        """

        messages, conversation_id = self.create_or_load_conversation(conversation_id, user_id, user_question)

        while True:
            print("asking the model")

            try:
                resp = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=self.get_available_tools(),
                    temperature=0.6,
                    top_p=0.95,
                    tool_choice="auto",
                    timeout=60,

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
                # args = first_call.function.arguments
                # parsed_args = json.loads(args)
                # tool_call = self.tool_call(first_call.id,first_call.function.name,parsed_args)
                # messages.append(self.assistant_message(tool_call=tool_call))
                messages.append(msg)
                self.insert_message_db(conversation_id, user_id, "assistant",
                                       content=f"tool_call: {first_call.function.name}",
                                       tool_call_id=first_call.id)

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
                                           tool_call_id=first_call.id)

                continue

            if msg.content:
                print("got normal message:", msg)
                continue
            else:
                return {"error": "there was an error communicating with the AI! please try again."}

    def stream_response(self, user_question: str, user_id: str, conversation_id: str = None):
        """
        Generator function that yields streaming responses from the agentic workflow.
        Designed to work with Server-Sent Events (SSE).
        """
        yield json.dumps({"message": "Processing your question...", "done": False})

        try:
            print("started streaming")
            messages, conversation_id = self.create_or_load_conversation(conversation_id, user_id, user_question)

            yield json.dumps({"message": "Analyzing available data sources...", "done": False})
            num_tries = 0  # Consider if num_tries should be reset under certain conditions or if it's for overall LLM communication

            while True:  # Main loop for interaction with the LLM
                try:
                    print("asking the model")
                    # Calculate num_tokens accurately based on your model's tokenizer if possible
                    # The current len(str(message))/4 is a rough estimate.
                    num_tokens = sum(len(str(m)) / 4 for m in messages)  # Rough estimate
                    print(
                        f"using {model_name} with {num_tokens} tokens")  # Replace model_name with self.model_name or actual variable
                    resp = self.groq_client.chat.completions.create(
                        model=model_name,  # Replace model_name
                        messages=messages,
                        tools=self.get_available_tools(),
                        temperature=0.6,
                        top_p=0.95,
                        tool_choice="required",  # This forces a tool call every time. Is this intended?
                        # If the model should sometimes respond directly, consider "auto".
                        timeout=60,
                    )
                    print("model answered")
                    yield json.dumps(
                        {"message": "Understanding user intention...", "done": False})  # More generic message
                except APITimeoutError as e:
                    yield json.dumps(
                        {"message": "Timeout communicating with AI. Please check internet connection.", "done": True})
                    return
                except APIConnectionError as e:
                    yield json.dumps(
                        {"message": "Connection error with AI. Please check internet connection.", "done": True})
                    return
                except RateLimitError as e:
                    yield json.dumps(
                        {"message": "Rate limit exceeded. Please upgrade or try again later.", "done": True})
                    return
                except Exception as e:  # Catch other model call exceptions
                    print(f"Error calling model: {e}")
                    print(messages)
                    yield json.dumps({"message": f"Error communicating with AI: {str(e)}", "done": True})
                    return  # Critical error with LLM call

                msg = resp.choices[0].message

                if msg.tool_calls:
                    first_call = msg.tool_calls[0]
                    function_name = first_call.function.name

                    try:
                        args_str = first_call.function.arguments
                        args = ast.literal_eval(args_str) if args_str else {}
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing tool arguments: {args_str}. Error: {e}")
                        # Inform LLM about parsing error and ask it to correct arguments format
                        error_content_for_llm = f"Invalid arguments format provided for tool {function_name}: {str(e)}. Arguments received: {args_str}. Please ensure arguments are a valid JSON string."
                        messages.append({
                            "tool_call_id": first_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": error_content_for_llm,
                        })
                        self.insert_message_db(conversation_id, user_id, "tool",
                                               content=f"Argument parsing error: {e}. Args: {args_str}",
                                               # Log detailed error
                                               tool_call_id=first_call.id)
                        yield json.dumps(
                            {"message": f"Error in arguments for {function_name}. AI will attempt to correct.",
                             "done": False})
                        continue  # Go back to LLM to retry formatting arguments

                    # 1. Append Assistant's decision to call the tool
                    # Assuming self.tool_call creates the appropriate dict structure for the 'tool_calls' field in an assistant message
                    assistant_tool_call_request = self.tool_call(first_call.id, function_name, args)
                    messages.append(self.assistant_message(message=msg.content or "",
                                                           tool_call=assistant_tool_call_request))  # msg.content might be None

                    yield json.dumps({"message": f"Using {function_name} to process your question...", "done": False})

                    tool_out = self.use_tools(msg.tool_calls)  # Execute the tool

                    # 2. Handle tool output and append ONE message for it to 'messages' for the LLM
                    tool_response_content_for_llm = ""
                    full_tool_output_for_db = str(tool_out)  # Default for DB logging

                    if function_name == "get_user_clarification":
                        # Assuming tool_out is like {"clarification": "What is X?"}
                        tool_response_content_for_llm = str(tool_out)  # Or json.dumps(tool_out)
                        messages.append({
                            "tool_call_id": first_call.id, "role": "tool", "name": function_name,
                            "content": tool_response_content_for_llm
                        })
                        self.insert_message_db(conversation_id, user_id, "tool", content=full_tool_output_for_db,
                                               tool_call_id=first_call.id)
                        print("got clarification question -> done")
                        yield json.dumps(
                            {"message": tool_out.get("clarification", "Please clarify your request."), "done": True})
                        return

                    elif function_name == "execute_sql":
                        sql_query_from_args = args.get('sql', "SQL query not found in arguments")
                        if isinstance(tool_out, pd.DataFrame):  # SQL Success (assuming pd is pandas alias)
                            yield json.dumps({"message": "Formatting database results...", "done": False})
                            summary = self._steps_summarize(sql_query_from_args, tool_out)
                            # For LLM: concise confirmation of success
                            tool_response_content_for_llm = f"SQL query executed successfully. A summary of the results is available."
                            # The actual data/summary for the user is yielded below.
                            # The LLM just needs to know the call succeeded.

                            messages.append({
                                "tool_call_id": first_call.id, "role": "tool", "name": function_name,
                                "content": tool_response_content_for_llm
                            })
                            # For DB & User: Log the assistant message that will contain the summary
                            self.insert_message_db(conversation_id, user_id, "assistant", tool_call_id=first_call.id,
                                                   content=summary, extracted_sql=sql_query_from_args,
                                                   reasoning=summary)
                            yield json.dumps({"message": summary, "sql": sql_query_from_args, "done": True})
                            print("got the final result -> done")
                            return
                        else:  # SQL Error
                            error_str = str(tool_out)
                            full_tool_output_for_db = error_str  # Already set, but for clarity
                            # For LLM: summarized error
                            summarized_error_for_llm = f"SQL execution failed. Error: {error_str[:250]}"  # Truncate for LLM history
                            tool_response_content_for_llm = summarized_error_for_llm

                            messages.append({
                                "tool_call_id": first_call.id, "role": "tool", "name": function_name,
                                "content": tool_response_content_for_llm
                            })
                            self.insert_message_db(conversation_id, user_id, "tool", content=full_tool_output_for_db,
                                                   tool_call_id=first_call.id)
                            yield json.dumps(
                                {"message": f"Error during database query. The AI will attempt to correct it.",
                                 "done": False})
                            # No num_tries increment here, as the LLM might fix it. Loop will continue.
                            continue  # Let LLM retry based on the summarized error
                    else:  # Other tools
                        # For LLM:
                        if isinstance(tool_out, Exception):  # If your use_tools can return Exception for other tools
                            tool_response_content_for_llm = f"Tool '{function_name}' failed. Error: {str(tool_out)[:250]}"
                        else:
                            tool_response_content_for_llm = str(tool_out)  # Or a summary if it can be very long
                            if len(tool_response_content_for_llm) > 500:  # Arbitrary limit for LLM history
                                tool_response_content_for_llm = tool_response_content_for_llm[:500] + "... (truncated)"

                        messages.append({
                            "tool_call_id": first_call.id, "role": "tool", "name": function_name,
                            "content": tool_response_content_for_llm
                        })
                        self.insert_message_db(conversation_id, user_id, "tool", content=full_tool_output_for_db,
                                               tool_call_id=first_call.id)
                        yield json.dumps(
                            {"message": f"Tool {function_name} processed your request. Analyzing results...",
                             "done": False})
                        continue  # Let LLM continue with the output of this other tool

                elif msg.content:  # Model responded with content instead of a tool call
                    # This might happen if tool_choice="auto" or if the model is trying to explain an issue.
                    # You might want to append this message to history and decide if a retry is needed or if it's a final answer.
                    messages.append({'role': 'assistant', 'content': msg.content})
                    # If you expect a tool call but got content, it might be an error or the model giving up on tools.
                    yield json.dumps({"message": f"AI response: {msg.content}",
                                      "done": False})  # Or True if this is considered final.
                    # Potentially, force a retry with a user message asking to use tools if that's the desired flow:
                    # messages.append(self.user_message("Please use the available tools to answer the question."))
                    # num_tries += 1 # if you consider this a failed attempt to use a tool
                    # yield json.dumps({"message": "The AI did not use a tool as expected. It will retry.", "done": False})
                    # print("AI provided content instead of tool call. Retrying...")
                    # continue

                    # For now, let's assume if we get content and tool_choice was "required", something is off.
                    # If tool_choice="required", this branch ideally shouldn't be hit unless the API behaves unexpectedly.
                    print(f"Model provided content when a tool call was required: {msg.content}")
                    yield json.dumps({"message": "AI provided an unexpected response. Retrying...", "done": False})
                    num_tries += 1  # Count this as a general attempt failure if a tool was expected
                    # continue # Go to next iteration to retry the LLM call

                # Fallback/Error handling for LLM not calling tools or providing content
                # This part is reached if msg.tool_calls is false AND msg.content is false/empty.
                # (Or if the above 'elif msg.content:' doesn't 'return' or 'continue' in all its paths)
                num_tries += 1
                if num_tries < 4:
                    yield json.dumps(
                        {'message': f"The AI is having trouble forming a response. Retrying {num_tries}/3...",
                         "done": False})
                    print(f"AI did not provide tool_calls or content. Retrying {num_tries}/3")
                    messages.append(self.user_message(
                        "Please attempt the previous operation again, ensuring you use a tool if appropriate."))  # Nudge the model
                    continue
                else:
                    yield json.dumps(
                        {
                            "message": "The AI is currently unable to process this request after multiple attempts. Please try again later or rephrase your question.",
                            "done": True})
                    print(
                        "Error communicating with the AI or AI failed to produce valid response -> done after retries")
                    return

        except Exception as e:
            print(f"Unhandled error in streaming response: {str(e)}")
            import traceback
            traceback.print_exc()
            yield json.dumps({"message": f"An unexpected error occurred: {str(e)}", "done": True})
            print("unexpected error while streaming -> done")