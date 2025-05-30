import requests
import pprint

pprint.pprint(requests.get("http://localhost:8084/api/v0/get_cache").json())