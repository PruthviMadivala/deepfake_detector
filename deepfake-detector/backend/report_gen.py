"""
Simple helper to save last_result.json used by Node /report route
"""
import json
from pathlib import Path
from datetime import datetime

OUT = Path(__file__).parent / "last_result.json"

def save_last_result(result_dict, filetype=None, timestamp=None):
    d = dict(result_dict)
    if filetype:
        d["filetype"] = str(filetype)
    if not timestamp:
        d["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        d["timestamp"] = timestamp
    OUT.write_text(json.dumps(d))
    return OUT
