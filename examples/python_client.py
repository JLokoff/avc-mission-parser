"""Run from repository root: python -m examples.python_client"""
from mission_parser import MissionParser

with MissionParser(model='granite350', quantization='Q8_0') as parser:
    for command in ['Collect the blue box in the northeast corner.', 'Pause the mission.']:
        result = parser.parse(command)
        print(result)
        if result['ok']:
            intent = result['intent']  # Pass to YOUR mission validator/state machine.
            # This example deliberately does not control actuators.
