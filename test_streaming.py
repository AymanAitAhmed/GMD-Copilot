import requests
import json
import time
import sseclient
import threading
from concurrent.futures import ThreadPoolExecutor

# Settings
BASE_URL = "http://127.0.0.1:8084"
TEST_QUESTION = "would say we are able to make 2000 bentos in one year?"
USER_ID = "user_id1012"


def initiate_query(question):
    """
    Step 1: Call the run_agent endpoint to start the process
    Returns the queryId needed for subsequent calls
    """
    url = f"{BASE_URL}/run_agent"
    payload = {"question": question}
    cookies = {"session_id": USER_ID}
    
    print(f"📤 Initiating query: '{question}'")
    try:
        response = requests.post(url, json=payload, cookies=cookies, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if "askingTask" in data and "queryId" in data["askingTask"]:
            query_id = data["askingTask"]["queryId"]
            status = data["askingTask"]["status"]
            print(f"✅ Query initiated successfully with ID: {query_id}")
            print(f"📊 Initial status: {status}")
            return query_id
        else:
            print("❌ Unexpected response format:", data)
            return None
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def poll_status(query_id, interval=1.0, max_polls=50):
    """
    Step 2: Poll the status endpoint to track progress
    Keep polling until the status is FINISHED or max_polls is reached
    """
    url = f"{BASE_URL}/api/ask_task/status"
    params = {"queryId": query_id}
    cookies = {"session_id": USER_ID}
    
    poll_count = 0
    
    print(f"🔍 Starting to poll status for query ID: {query_id}")
    
    while poll_count < max_polls:
        try:
            response = requests.get(url, params=params, cookies=cookies, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "askingTask" in data and "status" in data["askingTask"]:
                status = data["askingTask"]["status"]
                print(f"📊 Status update #{poll_count+1}: {status}")
                
                # Exit if we reached a terminal state
                if status in ["FINISHED", "FAILED", "STOPPED"]:
                    print(f"✅ Task completed with status: {status}")
                    return data
                    
            else:
                print("❌ Unexpected status response format:", data)
                
            poll_count += 1
            time.sleep(interval)
            
        except requests.RequestException as e:
            print(f"❌ Status polling failed: {e}")
            poll_count += 1
            time.sleep(interval)
    
    print("⚠️ Maximum polls reached without completion")
    return None


def listen_to_stream(query_id):
    """
    Step 3: Connect to the streaming endpoint to receive real-time updates
    """
    url = f"{BASE_URL}/api/ask_task/streaming?queryId={query_id}"
    headers = {"Accept": "text/event-stream"}
    cookies = {"session_id": USER_ID}
    
    print(f"🔌 Connecting to streaming endpoint for query ID: {query_id}")
    
    try:
        # Connect to the stream
        response = requests.get(url, headers=headers, cookies=cookies, stream=True, timeout=60)
        response.raise_for_status()
        
        print("🎬 Stream connected, waiting for messages...")
        
        # Manual parsing of the event stream
        buffer = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive chunks
                buffer += chunk.decode('utf-8')
                
                # Process complete events (separated by double newlines)
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    
                    # Extract the data part
                    for line in event.split("\n"):
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            try:
                                data = json.loads(data_str)
                                done = data.get("done", False)
                                message = data.get("message", "")
                                
                                print("\n📢 Stream update:")
                                print("=" * 50)
                                print(message)
                                print("=" * 50)
                                
                                if done:
                                    print("✅ Stream completed")
                                    return
                                    
                            except json.JSONDecodeError as e:
                                print(f"❌ Invalid JSON in stream: {data_str} - {e}")
        
    except requests.RequestException as e:
        print(f"❌ Stream connection failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error in stream: {e}")
        import traceback
        traceback.print_exc()

def run_full_test(question):
    """
    Run a complete test of all endpoints:
    1. Initiate the query
    2. Start streaming in a separate thread
    3. Poll the status to completion
    """
    query_id = initiate_query(question)
    if not query_id:
        print("❌ Test failed: Could not initiate query")
        return
    
    # Start streaming in a background thread
    stream_thread = threading.Thread(target=listen_to_stream, args=(query_id,))
    stream_thread.daemon = True
    stream_thread.start()
    
    # Wait a moment to let the stream connect
    time.sleep(1)
    
    # Poll the status in the main thread
    result = poll_status(query_id)
    
    # Wait for the streaming thread to finish (timeout after 30 seconds)
    stream_thread.join(timeout=30)
    
    if result:
        print("\n✅ Test completed successfully!")
    else:
        print("\n⚠️ Test completed with issues")


if __name__ == "__main__":
    # Check if sseclient is installed
    try:
        import sseclient
    except ImportError:
        print("⚠️ sseclient not found. Installing...")
        import pip
        pip.main(['install', 'sseclient-py'])
        import sseclient
    
    # Run the full test
    print("🧪 Starting full endpoint test...")
    run_full_test(TEST_QUESTION)
