"""Run after scripts/serve.py: python examples/http_client.py"""
import json
import urllib.request
req = urllib.request.Request('http://127.0.0.1:8080/parse',
    data=json.dumps({'text': 'Collect the blue box in the northeast corner.'}).encode(),
    headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as response:
    print(json.dumps(json.load(response), indent=2))
