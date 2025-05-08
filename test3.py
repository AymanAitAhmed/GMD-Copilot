import requests
import pprint
import json


pprint.pprint(json.loads(requests.get("http://localhost:8084/api/v0/load_question?id=conversation_id").content))