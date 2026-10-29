"""Small helpers shared by the paper cards (paper_cards.py), the comparison corpora (scoring_corpora.py), the judges (scoring_judges.py) and the evaluation of the forum proposals
(forum_scoring.py): hashes, append-only JSONL files, the subset of JSON Schema the answers are checked with, and the text normalisation used for exact-quotation checks.

Nothing here opens a database, reads `.env` or calls a model.
"""
import copy
import hashlib
import hmac
import json
import os
import re
import unicodedata
from pathlib import Path


# --- hashes and files ----------------------------------------------------------------------------------------------------------------------

def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_obj(obj):
    return sha256_text(obj if isinstance(obj, str) else canonical(obj))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, record):
    """Append one record and flush it to disk: these files are only ever appended to, never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json_once(path, obj):
    """Create a JSON file that must not exist yet (FileExistsError otherwise): a corpus manifest is written once and never revised."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


# --- JSON answers -------------------------------------------------------------------------------------------------------------------------

_STRIP = ("$schema", "$comment", "title", "minimum", "maximum")      # keywords a strict structured-output endpoint may not accept; check_json_schema enforces them instead


def provider_schema(schema):
    """The schema as sent to a provider with strict structured output: bounds and annotations removed (see _STRIP), everything else identical."""
    def clean(node):
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k not in _STRIP}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node
    return clean(copy.deepcopy(schema))


def _type_ok(value, expected):
    return {"string": isinstance(value, str), "boolean": isinstance(value, bool), "null": value is None, "object": isinstance(value, dict), "array": isinstance(value, list),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool)}[expected]


def check_json_schema(value, schema, path="$"):
    """Problems of `value` against the subset of JSON Schema the answers use: type (or list of types), enum, required, properties, additionalProperties false, items, minimum, maximum."""
    errors = []
    expected = schema.get("type")
    if expected is not None:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(value, t) for t in options):
            return [f"{path}: expected {'/'.join(options)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing {key!r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: unexpected key {key!r}")
        for key, sub in props.items():
            if key in value:
                errors += check_json_schema(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += check_json_schema(item, schema["items"], f"{path}[{i}]")
    return errors


def parse_json(text):
    """The model's text as JSON -> (object, None), or (None, reason)."""
    try:
        return json.loads(text), None
    except (TypeError, ValueError):
        return None, "the answer is not valid JSON"


# --- text ---------------------------------------------------------------------------------------------------------------------------------

INVISIBLE = dict.fromkeys([0x200b, 0x200c, 0x200d, 0x2060, 0xfeff, 0x00ad], None)
EVIDENCE_MARK = re.compile(r"\s*\[(?:[EPAW]\d+(?:\s*[,;]\s*[EPAW]?\d+)*)\]")      # the forum's internal evidence ids: [E3], [P1, P2]...


def normalize_text(text):
    """Unicode and whitespace only (ligatures such as ﬁ become fi, invisible characters go, runs of blanks collapse); paragraph breaks are kept."""
    text = unicodedata.normalize("NFKC", str(text or "")).translate(INVISIBLE).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_proposal_text(text):
    """A section of a forum proposal without the markers of the forum's internal evidence records, which a reader of the proposal cannot see."""
    return normalize_text(EVIDENCE_MARK.sub("", str(text or "")))


def squash_alnum(text):
    """Letters and digits only, case-folded: a quotation is compared with the text this way, because text extracted from a PDF has spurious spaces and hyphens ('profile - types')."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(text or "")).casefold())


def title_key(title):
    """A journal or paper title reduced to lower-case words, for matching names written differently."""
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", str(title or "")).casefold()))


def opaque_id(salt, key, prefix="in"):
    """A stable id that reveals nothing of what it names: HMAC-SHA256 of the key under a salt."""
    return f"{prefix}_" + hmac.new(str(salt).encode("utf-8"), str(key).encode("utf-8"), hashlib.sha256).hexdigest()[:16]
