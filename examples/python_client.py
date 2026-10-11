"""
Load a local model through the Python interface, interpret two example commands, and
print the results. Run python -m examples.python_client from the repository root. This
demonstrates keeping one model loaded for several commands; it does not issue any flight
commands.
"""
from mission_parser import MissionParser

with MissionParser(model='granite350', quantization='Q8_0') as parser:
    for command in ['Collect the blue box in the northeast corner.', 'Pause the mission.']:
        result = parser.parse(command)
        print(result)
        if result['ok']:
            intent = result['intent']  # Pass to YOUR mission validator/state machine.
            # This example deliberately does not control actuators.
