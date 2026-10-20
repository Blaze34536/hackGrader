"""Representative human and generated repository dumps for detector tests.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

HAND_WRITTEN = r'''REPOSITORY: student/notes-app
DEFAULT BRANCH: main
FILES INCLUDED (2): app.py, helpers.py

--- FILE: app.py ---
import os
from helpers import slugify

DATA = os.path.join(os.path.dirname(__file__), "notes")

def load(name):
    print("loading", name)
    with open(os.path.join(DATA, slugify(name))) as handle:
        return handle.read()

def save(name, body):
    # TODO: handle a full disk
    with open(os.path.join(DATA, slugify(name)), "w") as handle:
        handle.write(body)

def main():
    print(load("First Note"))
    save("second note", "hello")
    try:
        os.remove(os.path.join(DATA, "nope"))
    except Exception:
        pass

--- FILE: helpers.py ---
import re

def slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
'''

GENERATED = r'''REPOSITORY: student/notes-app
DEFAULT BRANCH: main
FILES INCLUDED (2): app.py, helpers.py

--- FILE: app.py ---
# ============ Notes Application ============
# ============ Notes Application ============

import os
from typing import Optional

DATA_DIRECTORY: str = "notes"

def load_note(note_name: str) -> Optional[str]:
    """Load note."""
    try:
        with open(os.path.join(DATA_DIRECTORY, note_name), "r") as handle:
            return handle.read()
    except Exception as exc:
        print(exc)
        return None

def save_note(note_name: str, body: str) -> bool:
    """Save note."""
    try:
        with open(os.path.join(DATA_DIRECTORY, note_name), "w") as handle:
            handle.write(body)
        return True
    except Exception as exc:
        print(exc)
        return False

def delete_note(note_name: str) -> bool:
    """Delete note."""
    try:
        os.remove(os.path.join(DATA_DIRECTORY, note_name))
        return True
    except Exception as exc:
        print(exc)
        return False

--- FILE: helpers.py ---
import re

def slugify(name: str) -> str:
    """Slugify name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
'''

FIXTURES = ((HAND_WRITTEN, "hand-written"), (GENERATED, "generated"))
