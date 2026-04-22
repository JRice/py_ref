"""
File I/O reference: JSON, CSV, YAML, TOML, XML.
All tests use tmp_path so nothing touches the real filesystem.
"""

import json
import csv
import yaml
import tomllib
import tomli_w
import xml.etree.ElementTree as ET
from pathlib import Path
from io import StringIO


# ===========================================================================
# JSON
# ===========================================================================

def test_json_write_and_read(tmp_path):
    data = {
        "users": [
            {"id": 1, "name": "Alice", "active": True},
            {"id": 2, "name": "Bob",   "active": False},
        ],
        "total": 2,
    }
    path = tmp_path / "users.json"
    path.write_text(json.dumps(data, indent=2))

    loaded = json.loads(path.read_text())
    assert loaded["total"] == 2
    assert loaded["users"][0]["name"] == "Alice"


def test_json_custom_encoder(tmp_path):
    from datetime import date

    class DateEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, date):
                return obj.isoformat()
            return super().default(obj)

    payload = {"event": "launch", "date": date(2025, 1, 15)}
    serialized = json.dumps(payload, cls=DateEncoder)
    assert '"2025-01-15"' in serialized


def test_json_streaming_large_array():
    # json.JSONDecoder can parse incrementally; simpler: ijson for huge files.
    # For moderate sizes, just use json.loads on the string.
    lines = [json.dumps({"n": i}) for i in range(5)]
    parsed = [json.loads(line) for line in lines]
    assert parsed[3]["n"] == 3


# ===========================================================================
# CSV
# ===========================================================================

def test_csv_write_and_read(tmp_path):
    path = tmp_path / "sales.csv"
    rows = [
        {"product": "Widget", "qty": 10, "price": 9.99},
        {"product": "Gadget", "qty": 5,  "price": 24.99},
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["product", "qty", "price"])
        writer.writeheader()
        writer.writerows(rows)

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        loaded = list(reader)

    assert len(loaded) == 2
    assert loaded[0]["product"] == "Widget"
    assert float(loaded[1]["price"]) == 24.99   # DictReader gives strings


def test_csv_from_string():
    raw = "name,score\nAlice,88\nBob,92\n"
    reader = csv.DictReader(StringIO(raw))
    rows = list(reader)
    assert rows[1]["name"] == "Bob"
    assert int(rows[0]["score"]) == 88


def test_csv_writer_quoting(tmp_path):
    path = tmp_path / "notes.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["description", "value"])
        writer.writerow(["price, with comma", 3.14])

    with path.open() as f:
        content = f.read()
    assert '"price, with comma"' in content


# ===========================================================================
# YAML
# ===========================================================================

def test_yaml_write_and_read(tmp_path):
    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "mydb",
        },
        "features": ["auth", "billing", "search"],
        "debug": False,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config, default_flow_style=False))

    loaded = yaml.safe_load(path.read_text())
    assert loaded["database"]["port"] == 5432
    assert "billing" in loaded["features"]
    assert loaded["debug"] is False


def test_yaml_multiline_strings():
    doc = yaml.safe_load("""
    message: |
      Line one.
      Line two.
    """)
    assert doc["message"] == "Line one.\nLine two.\n"


def test_yaml_multiple_documents():
    raw = "---\nname: Alice\n---\nname: Bob\n"
    docs = list(yaml.safe_load_all(raw))
    assert len(docs) == 2
    assert docs[1]["name"] == "Bob"


# ===========================================================================
# TOML  (tomllib reads, tomli-w writes)
# ===========================================================================

def test_toml_write_and_read(tmp_path):
    config = {
        "tool": {
            "name": "py-ref",
            "version": "0.1.0",
        },
        "dependencies": ["pytest", "httpx"],
        "build": {"optimize": True, "level": 2},
    }
    path = tmp_path / "pyproject.toml"
    path.write_bytes(tomli_w.dumps(config).encode())

    loaded = tomllib.loads(path.read_text())
    assert loaded["tool"]["name"] == "py-ref"
    assert "httpx" in loaded["dependencies"]
    assert loaded["build"]["level"] == 2


def test_toml_from_string():
    raw = """
    [server]
    host = "0.0.0.0"
    port = 8080
    workers = 4
    """
    cfg = tomllib.loads(raw)
    assert cfg["server"]["port"] == 8080
    assert cfg["server"]["workers"] == 4


# ===========================================================================
# XML — xml.etree.ElementTree
# ===========================================================================

def test_xml_write_and_read(tmp_path):
    # Build a tree
    root = ET.Element("catalog")
    for name, price in [("Widget", "9.99"), ("Gadget", "24.99")]:
        item = ET.SubElement(root, "item")
        ET.SubElement(item, "name").text = name
        ET.SubElement(item, "price").text = price

    tree = ET.ElementTree(root)
    path = tmp_path / "catalog.xml"
    ET.indent(tree, space="  ")    # Python 3.9+ pretty-printing
    tree.write(path, encoding="unicode", xml_declaration=True)

    # Parse it back
    loaded = ET.parse(path).getroot()
    items = loaded.findall("item")
    assert len(items) == 2
    assert items[0].find("name").text == "Widget"
    assert float(items[1].find("price").text) == 24.99


def test_xml_fromstring_and_xpath():
    xml = """<feed>
      <entry id="1"><title>Hello World</title><tag>python</tag></entry>
      <entry id="2"><title>Second Post</title><tag>web</tag></entry>
    </feed>"""
    root = ET.fromstring(xml)

    titles = [e.text for e in root.findall(".//title")]
    assert titles == ["Hello World", "Second Post"]

    # XPath attribute predicate
    entry1 = root.find(".//entry[@id='1']")
    assert entry1.find("tag").text == "python"


def test_xml_namespace():
    xml = """<root xmlns:dc="http://purl.org/dc/elements/1.1/">
      <dc:title>My Document</dc:title>
    </root>"""
    root = ET.fromstring(xml)
    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    title = root.find("dc:title", ns)
    assert title.text == "My Document"


# ===========================================================================
# pathlib extras — useful alongside file I/O
# ===========================================================================

def test_pathlib_operations(tmp_path):
    # Build a directory tree
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "src" / "util.py").write_text("def noop(): pass")
    (tmp_path / "README.md").write_text("# project")

    # Glob
    py_files = sorted(tmp_path.rglob("*.py"))
    assert len(py_files) == 2
    assert py_files[0].name == "main.py"

    # Read / suffix / stem
    f = py_files[0]
    assert f.suffix == ".py"
    assert f.stem == "main"
    assert f.read_text() == "print('hello')"

    # Rename
    new_path = f.with_name("app.py")
    f.rename(new_path)
    assert new_path.exists()
    assert not f.exists()
