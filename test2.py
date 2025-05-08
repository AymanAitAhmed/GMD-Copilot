import requests

def call_run_agent(server_url: str, question: str):
    """
    Calls the Flask /run_agent endpoint with the given question
    and prints out the SQL (or clarification) returned.
    """
    url = f"{server_url.rstrip('/')}/run_agent"
    payload = {"question": question}
    cookies = {"session_id": "user_id1012"}
    try:
        r = requests.post(url, json=payload, cookies=cookies,timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        print("Request failed:", e)
        return

    data = r.json()
    if "success" in data:
        print("Agent response:\n", data["success"])
    else:
        print("Unexpected response:", data)

if __name__ == "__main__":
    # adjust host/port as needed
    call_run_agent("http://127.0.0.1:8084", "would say we are able to make 2000 bentos in one year?")
