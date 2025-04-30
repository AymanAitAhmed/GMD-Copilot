import re


def extract_sql(llm_response: str) -> str:
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
        return sql

    # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
    sqls = re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
    if sqls:
        sql = sqls[-1]
        return sql

    # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
    sqls = re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
    if sqls:
        sql = sqls[-1]
        return sql

    sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
    if sqls:
        sql = sqls[-1]
        return sql

    return ""

from openai import OpenAI

api_key = "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"
OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1/",temperature=0.1)

print(extract_sql("SELECT COUNT(*) AS annual_bento_production FROM table_states WHERE state = '35' AND table_name = 'TAB10' AND EXTRACT(YEAR FROM start_time) = 2024;"))