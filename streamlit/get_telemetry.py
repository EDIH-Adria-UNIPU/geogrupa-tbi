import json
import tempfile
import telemetry_parser

def telemetry(video):
    video.seek(0)  

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.read())
        tmp_path = tmp.name

    rec = telemetry_parser.Parser(tmp_path).telemetry()[0]

    with open("telemetry_835.json", "w") as f:
        json.dump(rec, f, indent=2)
