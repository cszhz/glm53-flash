"""Load a torch-tpu `.pt.trace.json.gz`, repairing unescaped quotes.

The exporter writes HLO text into `"name"` (and some `args`) values verbatim,
so `custom_call_target="X64SplitHigh"` arrives with bare inner quotes and the
file is not valid JSON. The writer emits one `"key": value` per line, so fix
each string-valued line by escaping every quote between the first and last.
"""
import gzip
import json
import re

_KV = re.compile(r'^(\s*"[^"]+": ")(.*)("(,?)\s*)$')


def _fix(line: str) -> str:
    m = _KV.match(line)
    if not m or '"' not in m.group(2):
        return line
    inner = m.group(2).replace('\\"', '"').replace('"', '\\"')
    return m.group(1) + inner + m.group(3)


def load_events(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        text = "".join(_fix(l) for l in f)
    blob = json.loads(text)
    return blob["traceEvents"] if isinstance(blob, dict) else blob
