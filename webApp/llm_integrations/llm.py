import os

from openai import OpenAI

from ..base.base import CopilotBase


class OpenAI_Chat(CopilotBase):
    def __init__(self, client=None, config=None):
        CopilotBase.__init__(self, config=config)

        # default parameters - can be overrided using config
        self.temperature = 0.3

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

    def tool_message(self, name, call_id, tool_out) -> any:
        return {
            "role": "tool",
            "name": name,
            "tool_call_id": call_id,
            "content": tool_out
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
