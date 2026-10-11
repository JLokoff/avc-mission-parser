"""
Send one example mission command to the already running local HTTP API and print its
response. Start step 06 first, then run python examples/http_client.py. This
demonstrates how a separate team program can use the parser without loading a model
itself.
"""
import json
import urllib.request
req = urllib.request.Request('http://127.0.0.1:8080/parse',
    data=json.dumps({'text': 'Collect the blue box in the northeast corner.'}).encode(),
    headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as response:
    print(json.dumps(json.load(response), indent=2))
