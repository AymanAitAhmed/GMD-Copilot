import pandas as pd

examples = pd.read_csv(r'.\data\few_shot_examples.csv').to_dict(orient="records")

documentation = [
    "our business defines a cycle as going from a state = 35 to the next state = 35, so like this 35,36,37,...,34,35",
    "our business considers a part has been manufactured in table 10 if a state=35 but in table 07 and table 03 its if state=30 has been registered in table_states",
    "Please follow these steps to calculate the OEE also know as TRS: Filter the Data: Only include records where state_description is one of ('Loading', 'Processing', 'Waiting', 'Welding', 'Unloading'). Consider only records where the start_time falls between 08:00:00 and 16:00:00. Exclude records from the current day. Calculate Duration per Record: Compute the duration (in seconds) for each record using the difference between end_time and start_time. Compute Median Duration per Day and State: For each day (extracted from start_time) and for each state_description, calculate the median of the duration_seconds. Calculate Downtime: For each record, if its duration_seconds is greater than the median duration for that day and state, calculate the “excess” duration as: ini Copy Edit downtime = duration_seconds - median_duration Otherwise, consider downtime as zero. Sum the downtime for each day. Then subtract a fixed value of 60600 seconds from this daily sum to get the total downtime for the day. Determine Production Data: Define production as the count of records where state is '35'. Count these records for each day, using a slightly broader time window (e.g., from 07:00:00 to 16:00:00), and again, exclude the current day. This gives you the total units produced per day. Quality Data: For this calculation, if a table of non conform parts is provided then (quality data = total units produced - non conform parts) and if not provided assume all produced units are good (i.e., good units = total units produced). Set Operating Time: Use a fixed operating time of 8 hours per day (which equals 28,800 seconds). Calculate TRS (OEE): Combine the above data in this formula: mathematica Copy Edit TRS = ((Operating Time - Total Downtime) / Operating Time) * ((Total Units Produced * 1100) / (Operating Time - Total Downtime)) * (Good Units Produced / Total Units Produced) Calculate this TRS value for each day. Output: Finally, output the daily TRS values, ordered by day.",
    "our business considers production in table 10 to be perfect if its 22 parts a day, acceptable if its 20 or 21 parts a day and not acceptable for anything below 20",
    "our business only works from 8AM to 17PM.",
]


api_key = "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"

logo_path = "https://i.ibb.co/FL3mpvWR/gmd-logo.png"

tmp_file_dir = "webApp/tmp/"

app_secret_key = b'_5#y2L"F4Q8z\n\xec]/'

chromadb_path = r"webApp\chromadb"


