import re


def replace_newlines(text, token_limit=5):

    pattern = r'\n(?:\s*[^\n]{0,' + str(token_limit - 1) + r'}\s*\n)+'
    text = re.sub(pattern, '<br><br>', text)
    text = text.replace('\n', '<br>')

    return text
# Example Usage
example_text = "\nThis is a test.\n\nAnother paragraph.\n<li>\nSome list item.\n\n\nEnd."
result = replace_newlines(example_text)
print("""WITH annual_capacity AS (\n SELECT 10000 AS annual_capacity\n),\nmachine_utilization AS (\n SELECT \n COUNT(*) AS total_units_produced,\n COUNT(DISTINCT table_name) AS total_machines\n FROM table_states\n WHERE state = '35'\n AND start_time >= DATE_TRUNC('year', CURRENT_DATE)\n AND start_time < DATE_TRUNC('year', CURRENT_DATE) + INTERVAL '1 year'\n),\nrequired_machines AS (\n SELECT \n CEIL(ac.annual_capacity / (mu.total_units_produced / mu.total_machines)) AS machines_needed\n FROM annual_capacity ac, machine_utilization mu\n)\nSELECT machines_needed FROM required_machines;""")
