import requests

payload = {
    "sql": "SELECT COUNT(*) FROM table_states WHERE state = 35 AND table_name = 'TAB10' AND date_part('month', end_time::TIMESTAMP) = '02' AND date_part('year', end_time::TIMESTAMP) = '2025';",
    "question": "how many parts were manifactured during february 2025?"
}

response = requests.get("http://localhost:8084/api/v0/generate_plotly_figure", params=payload)
print(response.json())
