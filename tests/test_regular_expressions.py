"""
Common Python regular expressions -- 201-level reference.
Each test is a self-contained example of a real-world use case.
"""

import re


# ---------------------------------------------------------------------------
# basic matching
# ---------------------------------------------------------------------------

def test_search_match_and_fullmatch():
    text = "Order #1234 shipped"

    # search finds the pattern anywhere in the string
    found = re.search(r"\d+", text)
    assert found is not None
    assert found.group() == "1234"

    # match only checks from the beginning
    assert re.match(r"Order", text).group() == "Order"
    assert re.match(r"\d+", text) is None

    # fullmatch requires the whole string to match the pattern
    assert re.fullmatch(r"\d{5}", "02139").group() == "02139"
    assert re.fullmatch(r"\d{5}", "zip 02139") is None


def test_raw_strings_keep_patterns_readable():
    # Prefer raw strings for regex patterns so backslashes stay literal.
    windows_path = r"C:\Users\alice\notes.txt"
    assert re.search(r"\\Users\\(\w+)\\", windows_path).group(1) == "alice"


# ---------------------------------------------------------------------------
# character classes and anchors
# ---------------------------------------------------------------------------

def test_character_classes_and_repetition():
    sku = "ABC-12345"

    assert re.fullmatch(r"[A-Z]{3}-\d{5}", sku)
    assert re.fullmatch(r"[A-Z]{3}-\d{5}", "abc-12345") is None

    words = re.findall(r"[A-Za-z]+", "milk, eggs, bread")
    assert words == ["milk", "eggs", "bread"]


def test_anchors_and_word_boundaries():
    log_line = "ERROR payment failed"

    assert re.search(r"^ERROR", log_line)
    assert re.search(r"failed$", log_line)

    text = "cat concatenate scatter"
    assert re.findall(r"\bcat\b", text) == ["cat"]


# ---------------------------------------------------------------------------
# extracting data
# ---------------------------------------------------------------------------

def test_capturing_groups():
    date_text = "created=2026-04-25"
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_text)

    assert match is not None
    assert match.group(0) == "2026-04-25"      # the whole match
    assert match.group(1) == "2026"            # first capture
    assert match.groups() == ("2026", "04", "25")


def test_named_capturing_groups():
    email = "Ada Lovelace <ada@example.com>"
    pattern = r"(?P<name>[\w ]+) <(?P<email>[\w.-]+@[\w.-]+)>"
    match = re.fullmatch(pattern, email)

    assert match is not None
    assert match.group("name") == "Ada Lovelace"
    assert match.groupdict() == {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
    }


def test_findall_for_simple_lists():
    text = "alice: 3, bob: 5, carol: 2"

    names = re.findall(r"[a-z]+", text)
    numbers = [int(number) for number in re.findall(r"\d+", text)]

    assert names == ["alice", "bob", "carol"]
    assert numbers == [3, 5, 2]


def test_finditer_for_match_objects_and_positions():
    text = "red green blue"
    matches = list(re.finditer(r"\w+", text))

    assert [match.group() for match in matches] == ["red", "green", "blue"]
    assert matches[1].span() == (4, 9)


# ---------------------------------------------------------------------------
# replacing and splitting
# ---------------------------------------------------------------------------

def test_sub_replaces_matching_text():
    phone = "Call 555-123-4567"
    redacted = re.sub(r"\d{3}-\d{3}-\d{4}", "[phone]", phone)

    assert redacted == "Call [phone]"


def test_sub_with_a_function():
    text = "Subtotal: $20, Tax: $2"

    def add_one(match: re.Match[str]) -> str:
        dollars = int(match.group(1))
        return f"${dollars + 1}"

    assert re.sub(r"\$(\d+)", add_one, text) == "Subtotal: $21, Tax: $3"


def test_split_on_flexible_delimiters():
    tags = "python, regex; testing | pytest"
    parts = re.split(r"\s*[,;|]\s*", tags)

    assert parts == ["python", "regex", "testing", "pytest"]


# ---------------------------------------------------------------------------
# compiled patterns and flags
# ---------------------------------------------------------------------------

def test_compile_for_reuse():
    email_pattern = re.compile(r"^[\w.-]+@[\w.-]+\.[a-z]{2,}$", re.IGNORECASE)

    assert email_pattern.fullmatch("Ada@example.COM")
    assert email_pattern.fullmatch("not an email") is None


def test_ignorecase_and_multiline_flags():
    text = "INFO boot\nerror disk full\nINFO shutdown"

    errors = re.findall(r"^error .+$", text, flags=re.IGNORECASE | re.MULTILINE)

    assert errors == ["error disk full"]


def test_verbose_patterns_for_readability():
    phone_pattern = re.compile(
        r"""
        ^\(?(\d{3})\)?    # area code, optionally wrapped in parentheses
        [-.\s]?           # optional separator
        (\d{3})           # prefix
        [-.\s]?           # optional separator
        (\d{4})$          # line number
        """,
        re.VERBOSE,
    )

    match = phone_pattern.fullmatch("(555) 123-4567")
    assert match is not None
    assert match.groups() == ("555", "123", "4567")


# ---------------------------------------------------------------------------
# useful pattern techniques
# ---------------------------------------------------------------------------

def test_greedy_vs_non_greedy_matching():
    html = "<b>first</b><b>second</b>"

    assert re.search(r"<b>.*</b>", html).group() == "<b>first</b><b>second</b>"
    assert re.search(r"<b>.*?</b>", html).group() == "<b>first</b>"


def test_optional_groups():
    urls = ["https://example.com", "http://localhost", "example.org"]
    pattern = re.compile(r"^(?:https?://)?([\w.-]+)$")

    assert [pattern.fullmatch(url).group(1) for url in urls] == [
        "example.com",
        "localhost",
        "example.org",
    ]


def test_lookaround_assertions():
    text = "prices: $10, $25, 30 EUR"

    dollar_amounts = re.findall(r"(?<=\$)\d+", text)
    numbers_not_followed_by_currency = re.findall(r"\b\d+\b(?! EUR)", text)

    assert dollar_amounts == ["10", "25"]
    assert numbers_not_followed_by_currency == ["10", "25"]


def test_escape_user_input_before_building_a_pattern():
    # re.escape treats user input as literal text instead of regex syntax.
    search_term = "python.org?"
    text = "Did you mean python.org? The dot is part of the name."

    pattern = re.compile(re.escape(search_term))

    assert pattern.search(text).group() == "python.org?"
