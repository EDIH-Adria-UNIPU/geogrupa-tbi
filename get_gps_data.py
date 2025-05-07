"""
Creates a JSON file with the telemetry data from the video.
"""

import json

import telemetry_parser

rec = telemetry_parser.Parser("220919_111808250.mp4").telemetry()[0]

with open("telemetry_250.json", "w") as f:
    json.dump(rec, f, indent=2)
