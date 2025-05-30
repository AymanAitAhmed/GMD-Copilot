import pprint

import requests
import json
import psycopg2
import pandas as pd
import altair as alt # Used by the exec'd code and for type hinting
import asyncio # For async LLM call

# --- Configuration ---
GRAFANA_URL = "http://localhost:3000"  # Your Grafana instance URL
GRAFANA_API_KEY = "glsa_cWZIoQFZ8ibLMQEzLoGau7wGcoI4kjem_39cf413b" # A Grafana API key with Editor/Admin role
DATABASE_PATH = "your_database.db" # Path to your SQLite database or connection string

# --- SQL Query ---
# This is the SQL query Grafana will use. Its columns are crucial for the LLM.
SQL_QUERY = "select count(state),state from table_states GROUP BY state"

# --- Grafana Dashboard and Panel Configuration ---
DASHBOARD_TITLE = "LLM-Powered Vega-Lite Dashboard"
PANEL_TITLE = "Dynamic Chart (from LLM)" # This might be overridden by LLM later
GRAFANA_DATASOURCE_NAME = "Your SQL Datasource Name in Grafana"

# --- LLM Configuration ---
# In a real scenario, you'd prompt the user or have a more complex way to define this.
USER_CHART_REQUEST = "Show the trend of 'value' over 'event_timestamp', colored by 'category'. Make it a line chart. Also, show 'intensity' perhaps as point size if possible."

# --- Altair Chart Configuration (Defaults, might be overridden by LLM) ---
ALTAIR_CHART_WIDTH = "container" # Use "container" for Grafana Vega panel responsiveness
ALTAIR_CHART_HEIGHT = 300


async def get_sql_column_info(db_path, query):
    """
    Fetches column names and a sample row to infer data types for the LLM.
    Returns a dictionary of {column_name: pandas_dtype_str}
    """
    conn = None
    try:
        db_host = '127.0.0.1'
        db_port = 5432
        db_user = 'gmd'
        db_password = '1234'
        db_name = 'gmd_iot'
        conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password,
                                 port='5432')
        # Fetch one row to get column names and infer types
        # Ensure the query is safe and doesn't fetch too much data
        safe_query = f"SELECT * FROM ({query}) AS subquery LIMIT 1"
        sample_df = pd.read_sql_query(safe_query, conn)

        if sample_df.empty:
            cursor = conn.execute(f"PRAGMA table_info(({query}))") # This is SQLite specific
            print("Warning: Could not fetch sample row to infer types. LLM might lack type info.")
            # Attempt to get columns from a query that returns no data
            cursor = conn.cursor()
            cursor.execute(query.replace("SELECT ", "SELECT ").replace(" FROM ", " LIMIT 0 FROM "))
            column_names = [desc[0] for desc in cursor.description]
            return {name: "object" for name in column_names} # Default to object/string

        column_types = {col: str(sample_df[col].dtype) for col in sample_df.columns}
        print(f"Column info: {column_types}")
        return column_types
    except Exception as e:
        print(f"Error fetching column info from SQL: {e}. Provide column info manually for LLM.")
        # Fallback: Manually define or return empty if critical
        return {} # Or raise error
    finally:
        if conn:
            conn.close()

async def generate_altair_code_via_llm(columns_with_types, user_request):
    """
    Simulates calling an LLM to generate Altair Python code.
    In a real implementation, this would use the Gemini API.
    """
    print("\n--- Generating Altair Code via LLM ---")
    prompt_parts = [
        "You are an expert in Python and the Altair data visualization library.",
        "Your task is to generate Python code that defines an Altair chart object.",
        "The chart will be rendered in Grafana, which will provide the data.",
        "The Python code should assign the Altair chart object to a variable named `chart`.",
        "Do NOT include any data loading (e.g., pandas.read_csv) or `chart.show()` calls.",
        "The chart should be defined to expect data, e.g., `alt.Chart().mark_...` not `alt.Chart(my_dataframe)...`.",
        "Use `alt.X`, `alt.Y`, `alt.Color`, etc., for encoding definitions.",
        "Ensure field names in encodings exactly match the provided column names.",
        "Make the chart interactive using `.interactive()`.",
        "\nAvailable data columns and their inferred pandas dtypes (use this to choose appropriate Altair types like :T, :Q, :N, :O):"
    ]
    for col, dtype in columns_with_types.items():
        altair_type_suggestion = ":Q" # Quantitative default
        if "datetime" in dtype or "timestamp" in dtype:
            altair_type_suggestion = ":T" # Temporal
        elif dtype == "object" or dtype == "string":
            altair_type_suggestion = ":N" # Nominal
        elif dtype == "bool":
            altair_type_suggestion = ":N" # or :O (Ordinal)
        prompt_parts.append(f"- `{col}` (dtype: {dtype}, suggest Altair type {altair_type_suggestion})")

    prompt_parts.append(f"\nUser's chart request: \"{user_request}\"")
    prompt_parts.append("\nGenerate only the Python code block for Altair. For example:")
    prompt_parts.append(
        "```python\n"
        "import altair as alt\n\n"
        "# Chart definition based on user request and columns\n"
        "chart = alt.Chart().mark_line().encode(\n"
        "    x=alt.X('some_column:T', title='Some Column Time'),\n"
        "    y=alt.Y('another_column:Q', title='Another Value')\n"
        ").interactive()\n"
        "```"
    )
    final_prompt = "\n".join(prompt_parts)

    # In a real scenario, you would make the API call here:
    # Initialize chat history
    chat_history = [{"role": "user", "parts": [{"text": final_prompt}]}]
    payload = {"contents": chat_history}
    api_key = "" # Gemini API key (leave as "" for Canvas environment)
    # Use gemini-2.0-flash as it's generally good for code and available
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    try:
        # Using 'requests' for simplicity here, but in a browser, you'd use fetch
        # For this script, we'll simulate the call
        # response = await fetch_from_llm_api(api_url, payload) # Your actual async fetch
        # For now, simulate a response:
        print("Simulating LLM API call...")
        await asyncio.sleep(1) # Simulate network latency

        # --- SIMULATED LLM RESPONSE ---
        # This is what the LLM *should* return: a Python code string.
        # Adjust this example based on SQL_QUERY columns and USER_CHART_REQUEST
        simulated_llm_output = """import altair as alt

chart = alt.Chart().mark_text(
    fontSize=60,
    align='center',
    baseline='middle'
).encode(
    # 'parts_count' should be the column name from your SQL query (aliased if needed)
    text=alt.Text('parts_count:N') # Or :Q if you want numeric formatting options
).properties(
    title='Parts Manufactured in February 2025', # This title is fine
    width=200,
    height=100
).configure_view(
    strokeWidth=0
).configure_title(
    fontSize=20,
    anchor='middle'
).interactive()"""
        # --- END SIMULATED LLM RESPONSE ---

        # Extract Python code if LLM wraps it in ```python ... ```
        if "```python" in simulated_llm_output:
            python_code = simulated_llm_output.split("```python")[1].split("```")[0].strip()
        else:
            python_code = simulated_llm_output.strip()

        return python_code

    except Exception as e:
        print(f"Error calling LLM API (or in simulation): {e}")
        # Fallback to a very basic default chart code
        return """
import altair as alt
chart = alt.Chart().mark_text(text='LLM Error').properties(width=200, height=50)
"""

async def convert_llm_altair_code_to_spec(
    altair_python_code,
    chart_title=PANEL_TITLE,
    width=ALTAIR_CHART_WIDTH,
    height=ALTAIR_CHART_HEIGHT
):
    """
    Executes the LLM-generated Altair Python code and converts the resulting chart to a Vega-Lite spec.
    WARNING: Uses exec(), which is a security risk with untrusted code.
    """
    print("\n--- Converting LLM Altair Code to Vega-Lite Spec ---")
    print("WARNING: Executing LLM-generated code. This is a security risk.")

    local_scope = {}
    # Provide 'alt' in the execution scope for the generated code
    exec_globals = {'alt': alt, '__builtins__': __builtins__}

    try:
        exec(altair_python_code, exec_globals, local_scope)
        altair_chart_object = local_scope.get('chart')

        if not altair_chart_object:
            raise ValueError("LLM code did not define a variable named 'chart', or it was None.")
        if not isinstance(altair_chart_object, (alt.TopLevelMixin)):
             raise ValueError(f"The 'chart' variable from LLM code is not a valid Altair chart object. Type: {type(altair_chart_object)}")


        # Apply common properties if not set by LLM, and ensure data is not embedded
        # The LLM should ideally set title and dimensions if requested, but we can override/default.
        final_chart = altair_chart_object.properties(
            title=chart_title, # Can be overridden by LLM if it sets a title
            width=width,
            height=height
        )

        # Convert to Vega-Lite dictionary, ensuring data is NOT embedded
        # The chart object from LLM should be defined without data (e.g., alt.Chart() not alt.Chart(df))
        vega_lite_spec = final_chart.to_dict(validate="vega-lite") # altair > 5.0

        # Double check no data values are embedded (Grafana provides data)
        if 'values' in vega_lite_spec.get('data', {}):
            print("Warning: LLM-generated chart spec had embedded data values. Removing.")
            del vega_lite_spec['data']['values']
        if 'url' in vega_lite_spec.get('data', {}):
            print("Warning: LLM-generated chart spec had a data URL. Removing.")
            del vega_lite_spec['data']['url']
        if 'data' not in vega_lite_spec or not vega_lite_spec.get('data'): # Ensure data source is defined for Grafana if spec is minimal
             vega_lite_spec['data'] = {'name': 'grafana'} # Default name Grafana Vega plugin might use


        print("Successfully converted LLM Altair code to Vega-Lite spec.")
        # print(json.dumps(vega_lite_spec, indent=2)) # For debugging
        return vega_lite_spec

    except Exception as e:
        import traceback

        print(f"Error executing LLM Altair code or converting to spec: {e}")
        # Fallback to an error message spec
        error_chart = alt.Chart(title="Error in LLM Chart Generation").mark_text(
            text=f"Error: {str(e)[:100]}...",  # Show a snippet of the error
            size=10
        ).properties(width=width, height=height)
        return error_chart.to_dict()


def get_grafana_datasource_uid(grafana_url, api_key, datasource_name):
    # (This function remains the same as in the previous version)
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    url = f"{grafana_url}/api/datasources/name/{datasource_name.strip()}"
    print(f"Fetching Grafana datasource UID for: '{datasource_name}' from URL: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        uid = response.json()["uid"]
        print(f"Found datasource '{datasource_name}' with UID: {uid}")
        return uid
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Grafana datasource '{datasource_name}': {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Grafana API Response Status: {e.response.status_code}")
            print(f"Grafana API Response Text: {e.response.text[:500]}")
        return None
    except KeyError:
        print(f"Error: Datasource '{datasource_name}' found but UID is missing in response. Full response: {response.json()}")
        return None


def create_or_update_grafana_dashboard_with_vega_panel(
    grafana_url, api_key, dashboard_title, panel_title, sql_query, vega_lite_spec,
    datasource_uid="femntec9koz5se"
):
    # (This function remains largely the same, ensures panel type is 'vega')
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Ensure the panel title in the JSON uses the one from the spec (if LLM set one)
    # or the default panel_title.
    current_panel_title = vega_lite_spec.get("title", panel_title)
    if isinstance(current_panel_title, dict) and "text" in current_panel_title: # Altair titles can be objects
        current_panel_title = current_panel_title["text"]

    print("=======================")
    pprint.pprint(json.dumps(vega_lite_spec))
    print("=======================")

    panel_json = {
        "type": "akdor1154-vega-panel",  # Official Grafana Vega plugin panel ID
        "title": current_panel_title,
        "datasource": {
            "uid": "femntec9koz5se",
            "type": "datasource", # This can often be omitted if UID is specific enough
        },
        "targets": [
            {
                "refId": "A",
                "datasource": {"uid": "femntec9koz5se"},
                "rawSql": sql_query,
                "format": "table", # Vega plugin works best with 'table' format
            }
        ],
        "gridPos": {"h": 15, "w": 24, "x": 0, "y": 0}, # Adjusted for potentially larger charts
        "options": {
            "spec": json.dumps(vega_lite_spec), # Vega-Lite spec as a JSON string
        }
    }
    # If width/height are "container", Grafana's Vega panel handles it.
    # If they are numeric, they are passed in the spec.

    dashboard_payload = {
        "dashboard": {
            "title": dashboard_title,
            "panels": [panel_json],
            "time": {"from": "now-6h", "to": "now"},
            "timepicker": {},
            "timezone": "browser",
            "refresh": "30s",
            "schemaVersion": 38, # Use a recent schema version
            "uid": None # Let Grafana generate for new, or set for existing
        },
        "overwrite": True,
        "folderId": 0, # General folder
    }

    try:
        print(f"\nAttempting to create/update dashboard: {dashboard_title}")
        # print("Dashboard Payload (partial):", json.dumps(dashboard_payload, indent=2)[:1000])
        response = requests.post(f"{grafana_url}/api/dashboards/db", headers=headers, data=json.dumps(dashboard_payload), timeout=20)
        response.raise_for_status()
        dashboard_info = response.json()
        print(f"Successfully created/updated dashboard: {dashboard_title}")
        print(f"Dashboard URL: {grafana_url.rstrip('/')}{dashboard_info.get('url')}")
        return dashboard_info
    except requests.exceptions.RequestException as e:
        print(f"Error creating/updating Grafana dashboard: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Grafana API Response Status: {e.response.status_code}")
            print(f"Grafana API Response Text: {e.response.text[:500]}") # Print more of the error
        return None

async def main():
    """
    Main function to orchestrate the process.
    """
    print("--- Script Starting: Grafana Dashboard with LLM-Generated Altair/Vega-Lite ---")

    # 1. Get SQL column information for the LLM
    sql_columns_with_types = await get_sql_column_info(DATABASE_PATH, SQL_QUERY)
    if not sql_columns_with_types:
        print("Could not determine SQL columns for LLM. Using fallback or stopping.")
        # Fallback: Define some expected columns manually if needed for testing LLM part
        # sql_columns_with_types = {'event_timestamp': 'datetime64[ns]', 'value': 'float64', 'category': 'object'}
        # return # Or stop if this info is critical

    # 2. Generate Altair Python code via LLM (Simulated)
    llm_altair_code = await generate_altair_code_via_llm(sql_columns_with_types, USER_CHART_REQUEST)

    # 3. Convert LLM-generated Altair code to Vega-Lite spec
    # The title from PANEL_TITLE is a default; LLM might generate its own title.
    vega_lite_spec = await convert_llm_altair_code_to_spec(
        llm_altair_code,
        chart_title=PANEL_TITLE, # Default title, LLM can override
        width=ALTAIR_CHART_WIDTH,
        height=ALTAIR_CHART_HEIGHT
    )

    if not vega_lite_spec or vega_lite_spec.get("mark", {}).get("type") == "text" and "Error" in vega_lite_spec.get("title", "") :
        print("Halting script as Vega-Lite specification generation failed or resulted in an error chart.")
        return

    # 5. Create/Update Grafana Dashboard
    await asyncio.to_thread(create_or_update_grafana_dashboard_with_vega_panel,
        GRAFANA_URL,
        GRAFANA_API_KEY,
        DASHBOARD_TITLE,
        PANEL_TITLE, # This is the panel title for Grafana UI, spec might have its own internal title
        SQL_QUERY,
        vega_lite_spec
    )

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    # Setup for running asyncio main
    # In some environments (like Jupyter), you might need `nest_asyncio`
    # or run `asyncio.run(main())` directly if no event loop is running.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        loop = None

    if loop and loop.is_running():
        print("Async event loop already running. Scheduling main task.")
        # This might happen in environments like Jupyter notebooks with an active loop
        # For simplicity, we'll just run it. If issues, consider `nest_asyncio`.
        # Or, if this is the main script entry, direct run is usually fine.
        asyncio.run(main()) # This might error if loop is already running from external source
                            # A more robust way for notebooks:
                            # task = asyncio.create_task(main())
    else:
        asyncio.run(main())