"""Streamlit Community Cloud entrypoint for the existing Copilot."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

# Streamlit executes from the repository root. Reuse the adjacent Copilot modules.
app_directory = str(Path(__file__).resolve().parents[1])
if app_directory not in sys.path:
    sys.path.insert(0, app_directory)

from copilot_state import session_database
from ui import main


if __name__ == "__main__":
    if "cloud_case_directory" not in st.session_state:
        st.session_state["cloud_case_directory"] = TemporaryDirectory(prefix="blackwing-session-")
    database = Path(st.session_state["cloud_case_directory"].name) / "cases.sqlite3"
    with session_database(database):
        main(cloud=True)
