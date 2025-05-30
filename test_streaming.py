#!/usr/bin/env python3
import json
import requests

# ——— Configuration ———
BASE_URL        = "http://192.168.7.99:8084"
CONVERSATION_ID = "c9ef4125-dd96-4029-95bd-5a530ebcdba4"
SESSION_ID      = "your-session-id-value"   # set to whatever your Flask session cookie is

# ——— Helper to iterate SSE lines ———
def sse_lines(response):
    """
    Given a requests.Response with stream=True, yield each 'data:' payload (as text).
    """
    buffer = ""
    for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
        buffer += chunk
        if buffer.endswith("\n\n"):
            # we've reached the end of an SSE event
            for line in buffer.splitlines():
                if line.startswith("data:"):
                    yield line[len("data:"):].strip()
            buffer = ""

def main():
    url = f"{BASE_URL}/api/v0/api/stream_answer?conversation_id={CONVERSATION_ID}"
    cookies = {"session_id": SESSION_ID}

    print(f"→ Connecting to {url} …")
    with requests.get(url, stream=True, cookies=cookies) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            raise RuntimeError(f"Expected text/event-stream, got {content_type!r}")

        # Iterate the SSE stream
        for data_str in resp.iter_lines():
            resp_json = json.loads(data_str)
            print(resp_json['content'], end='')

if __name__ == "__main__":
    main()
