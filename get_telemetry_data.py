import json

import telemetry_parser

rec = telemetry_parser.Parser("220919_111512835.mp4").telemetry()[0]

with open("telemetry_835.json", "w") as f:
    json.dump(rec, f, indent=2)
