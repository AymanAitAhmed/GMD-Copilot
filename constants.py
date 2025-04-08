import pandas as pd

# examples = [
#     {'question': 'How many times does state 35 repeat within the period 2025-02-01 to 2025-02-05?',
#      'sql': "SELECT COUNT(*) FROM table_states WHERE state = 35 AND start_time >= '2025-02-01' AND end_time <= '2025-02-05'"},
#     {'question': 'What is the total number of states within the period 2025-03-01 to 2025-03-07?',
#      'sql': "SELECT COUNT(DISTINCT state) FROM table_states WHERE start_time >= '2025-03-01' AND end_time <= '2025-03-07'"},
#     {
#         'question': 'What is the average duration of execution of state 10 between the period 2025-02-01 to 2025-02-05 in hours?',
#         'sql': "SELECT AVG(EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp)) /60  AS period FROM table_states WHERE start_time >= '2025-02-01' AND end_time <= '2025-02-05' and state = '10';"},
#     {'question': 'combien de fois state 35 est repeter dans la period 2025-02-01 et 2025-02-05?',
#      'sql': "SELECT COUNT(*) FROM table_states WHERE state = '35' AND start_time >= '2025-02-01' AND end_time <= '2025-02-05'"},
#     {'question': 'quelle est la duree median de state welding en minutes pour tab10?',
#      'sql': "WITH durations AS (SELECT EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp) / 60 AS duration FROM tab10 WHERE state_description = 'Welding')\nselect PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration) AS median_duration_seconds from durations;"},
#     {'question': "quelle est la duree moyenne d'execution de state loading en minutes?",
#      'sql': "SELECT AVG(EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp)) / 60 AS period FROM tab10 WHERE state_description = 'Loading';"},
#     {
#         'question': "quelle est la duree total d'execution de state loading en minutes dans la period 2025-01-29 08:11:00 et 2025-01-29 15:29:00 pour tab07 ?",
#         'sql': "SELECT SUM(EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp)) / 3600 AS period FROM tab07 WHERE state_description = 'Loading' AND start_time >= '2025-01-29 08:11:00' AND end_time <= '2025-01-29 15:29:00';"},
#     {'question': 'donner moi 10 exemples de duree de welding dans tab10?',
#      'sql': "WITH durations AS (SELECT EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp) / 3600 AS duration FROM tab10 WHERE state_description = 'Welding') SELECT * FROM durations LIMIT 10;"},
#     {'question': 'combien de fois state Unloading est repeter dans tab03 dans la period 2025-02-01 et 2025-02-05?',
#      'sql': "SELECT COUNT(*) FROM tab03 WHERE state_description = 'Unloading' AND start_time >= '2025-02-01' AND end_time <= '2025-02-05'"},
#     {'question': 'how many parts were manifactured in the period 2025-02-01 to 2025-02-11?',
#      'sql': "SELECT count(*) FROM table_states WHERE state = '35' AND table_name = 'TAB10' AND start_time >= '2025-02-01' AND end_time <= '2025-02-11';"},
#     {'question': 'how many parts were manifactured during february 2025?',
#      'sql': "SELECT count(*) FROM table_states WHERE state = '35' AND table_name = 'TAB10' AND to_char(end_time::timestamp, 'MM') = '02' AND to_char(end_time::timestamp, 'YYYY') = '2025';"},
#     {'question': 'quelle est la duree median de chaque state dans tab10',
#      'sql': 'WITH durations AS (SELECT state_description, EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp) / 60 AS duration FROM tab10) select state_description, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration) AS median_duration_seconds from durations GROUP BY state_description;'},
#     {
#         'question': 'quelle est le downtime de chaque state dans tab10 en minutes dans la periode 2025-02-01 et 2025-02-03?',
#         'sql': 'WITH durations AS (\n                    SELECT start_time,state_description, EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp) / 60 AS duration \n                    FROM tab10),\n                    median_durations as (select state_description, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration) AS median_duration from durations GROUP BY state_description)\n                    ,downtimes AS (\n                      SELECT \n                        d.start_time,\n                        d.state_description,\n                        CASE \n                          WHEN d.duration > m.median_duration THEN d.duration - m.median_duration\n                          ELSE 0\n                        END AS downtime\n                      FROM durations d\n                      JOIN median_durations m on\n                        d.state_description = m.state_description\n                    )\n                    select * from downtimes;'},
#     {'question': 'quelle est la duree median de chaque state dans tab10 en minutes?',
#      'sql': 'WITH durations AS (SELECT state_description, EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp) / 60 AS duration FROM tab10)\nSELECT state_description, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration) AS median_duration_minutes FROM durations GROUP BY state_description;'},
#     {
#         'question': "give me the downtime of each state then Determine Production Data such that:  Define production as the count of records where state_description is 'Welding'. Count these records for each day, using a slightly broader time window (e.g., from 07:00:00 to 16:00:00), and again, exclude the current day. This gives you the total units produced per day.",
#         'sql': "WITH durations AS (SELECT start_time,state_description, EXTRACT(EPOCH FROM end_time::timestamp - start_time::timestamp) / 60 AS duration FROM tab10), median_durations as (select state_description, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration) AS median_duration from durations GROUP BY state_description), downtimes AS (SELECT d.start_time, d.state_description, CASE WHEN d.duration > m.median_duration THEN d.duration - m.median_duration ELSE 0 END AS downtime FROM durations d JOIN median_durations m on d.state_description = m.state_description), production_data AS (SELECT DATE(start_time) AS production_date, COUNT(*) AS units_produced FROM tab10 WHERE state_description = 'Welding' AND start_time::time >= '07:00:00' AND end_time::time <= '16:00:00' GROUP BY DATE(start_time)) SELECT * FROM production_data;"},
#     {'question': "what is the median duration of a cycle(from the first state 35 to the next one 35,36,...,35)?",
#      'sql': "WITH welding_states AS (\n    SELECT \n        id, \n        start_time, \n        end_time, \n        LEAD(start_time) OVER (ORDER BY start_time) AS next_welding_start_time\n    FROM table_states\n    WHERE state = '35'\n),\ncycle_durations AS (\n    SELECT \n        EXTRACT(EPOCH FROM next_welding_start_time - end_time) / 60 AS cycle_duration_minutes\n    FROM welding_states\n)\nSELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cycle_duration_minutes) AS median_cycle_duration_minutes\nFROM cycle_durations;"},
#     {'question': 'what is the duration of each cycle in minutes(a cycle is from state 35 to the next one 35,36,...,35)',
#      'sql': "WITH welding_states AS (\n    SELECT \n        id, \n        start_time, \n        end_time, \n        LEAD(start_time) OVER (ORDER BY start_time) AS next_welding_start_time,\n        DATE(start_time) AS cycle_day\n    FROM table_states\n    WHERE state = '35'\n),\ncycle_durations AS (\n    SELECT \n        cycle_day,\n        EXTRACT(EPOCH FROM next_welding_start_time - end_time) / 60 AS cycle_duration_minutes\n    FROM welding_states\n    WHERE next_welding_start_time IS NOT NULL\n)\nSELECT cycle_day, cycle_duration_minutes\nFROM cycle_durations;"},
#     {
#         'question': 'give me the the duration of each cycle in minutes, start_time of the cycle and end_time of the cycle(a cycle is from state 35 to the next one 35,36,...,35)',
#         'sql': "WITH welding_states AS (\n    SELECT \n        id, \n        start_time, \n        end_time, \n        LEAD(start_time) OVER (ORDER BY start_time) AS next_welding_start_time\n    FROM table_states\n    WHERE state = '35'\n),\ncycle_durations AS (\n    SELECT \n        start_time AS cycle_start_time,\n        next_welding_start_time AS cycle_end_time,\n        EXTRACT(EPOCH FROM next_welding_start_time - end_time) / 60 AS cycle_duration_minutes\n    FROM welding_states\n    WHERE next_welding_start_time IS NOT NULL\n)\nSELECT cycle_start_time, cycle_end_time, cycle_duration_minutes\nFROM cycle_durations;"}
# ]

examples = pd.read_csv(r'C:\Users\ACER\PycharmProjects\LogisticRegression\env\text2sql\data\few_shot_examples.csv').to_dict(orient="records")

documentation = [
    "our business defines a cycle as going from a state = 35 to the next state = 35, so like this 35,36,37,...,34,35",
    "our business considers a part has been manufactured in table 10 if a state=35 but in table 07 and table 03 its if state=30 has been registered in table_states",
    "Please follow these steps to calculate the OEE also know as TRS: Filter the Data: Only include records where state_description is one of ('Loading', 'Processing', 'Waiting', 'Welding', 'Unloading'). Consider only records where the start_time falls between 08:00:00 and 16:00:00. Exclude records from the current day. Calculate Duration per Record: Compute the duration (in seconds) for each record using the difference between end_time and start_time. Compute Median Duration per Day and State: For each day (extracted from start_time) and for each state_description, calculate the median of the duration_seconds. Calculate Downtime: For each record, if its duration_seconds is greater than the median duration for that day and state, calculate the “excess” duration as: ini Copy Edit downtime = duration_seconds - median_duration Otherwise, consider downtime as zero. Sum the downtime for each day. Then subtract a fixed value of 60600 seconds from this daily sum to get the total downtime for the day. Determine Production Data: Define production as the count of records where state is '35'. Count these records for each day, using a slightly broader time window (e.g., from 07:00:00 to 16:00:00), and again, exclude the current day. This gives you the total units produced per day. Quality Data: For this calculation, if a table of non conform parts is provided then (quality data = total units produced - non conform parts) and if not provided assume all produced units are good (i.e., good units = total units produced). Set Operating Time: Use a fixed operating time of 8 hours per day (which equals 28,800 seconds). Calculate TRS (OEE): Combine the above data in this formula: mathematica Copy Edit TRS = ((Operating Time - Total Downtime) / Operating Time) * ((Total Units Produced * 1100) / (Operating Time - Total Downtime)) * (Good Units Produced / Total Units Produced) Calculate this TRS value for each day. Output: Finally, output the daily TRS values, ordered by day.",

]


api_key = "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"

logo_path = "https://i.ibb.co/FL3mpvWR/gmd-logo.png"

tmp_file_dir = "C:/Users/ACER/PycharmProjects/LogisticRegression/env/text2sql/webApp/tmp/"

app_secret_key = b'_5#y2L"F4Q8z\n\xec]/'

chromadb_path = r"C:\Users\ACER\PycharmProjects\LogisticRegression\env\text2sql\webApp\chromadb"


