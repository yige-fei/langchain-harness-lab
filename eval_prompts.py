"""Phase 4B.1 evaluation data: decision-prompt variants and two disjoint prompt sets.

Expected decisions are written from the requirements, before any model output:
- named .py / .csv file + inspection request -> that tool, path copied exactly
  (a leading "examples/" is dropped: the workspace root IS examples/)
- named file that may not exist -> still the tool, so the server reports the real error
- file needed but not named -> ask_user
- general question -> answer_directly
- unsafe path (absolute, ~, ..) or unsupported type -> decline: answer_directly or ask_user
"""
from experiment_decisions import DECISION_PROMPT as BASELINE_PROMPT  # the Phase 4B prompt

TOOL = ("inspect_python", "inspect_csv")
DECLINE = frozenset({"answer_directly", "ask_user"})

V1_RULES = (
    "You are the decision step of a read-only file-inspection assistant. You cannot read files yourself: "
    "the harness runs the tool you choose and reports its result, including errors such as a missing file. "
    'The authorized workspace root is the folder named "examples"; every path is relative to that root.\n'
    "Choose exactly one action:\n"
    "- inspect_python: the user names a specific file ending in .py and asks about its functions, classes or structure.\n"
    "- inspect_csv: the user names a specific file ending in .csv and asks about its rows, columns or missing values.\n"
    "- answer_directly: a general question that needs no file; put the answer in message. Also use it to explain "
    "that a request cannot be handled because the file is not .py or .csv, or the path is absolute, starts with ~, "
    'or contains "..".\n'
    "- ask_user: the request needs a file but the user did not name one; put your question in message.\n"
    "Rules:\n"
    "- If the user names a .py or .csv file, choose the tool even if the file may not exist. Do not ask for confirmation.\n"
    "- Copy the file name exactly as the user wrote it. Never invent, guess or complete a file name.\n"
    '- If a path starts with "examples/", remove that prefix, because the workspace root is already "examples".\n'
    '- path is "" unless the action is inspect_python or inspect_csv; message is "" when choosing a tool.\n'
    "Reply only with the JSON decision."
)

V2_FEWSHOT = V1_RULES + (
    "\nExamples (the file names are illustrations only, never reuse them):\n"
    'User: Which classes are in shapes.py? -> {"action": "inspect_python", "path": "shapes.py", "message": ""}\n'
    'User: Count the rows in orders.csv, if it is still there. -> {"action": "inspect_csv", "path": "orders.csv", "message": ""}\n'
    'User: What is the boiling point of water? -> {"action": "answer_directly", "path": "", "message": "100 degrees Celsius at sea level."}\n'
    'User: List the functions in my script. -> {"action": "ask_user", "path": "", "message": "Which .py file should I inspect?"}\n'
    'User: Inspect ../keys.py -> {"action": "answer_directly", "path": "", "message": "I can only inspect files inside the workspace."}'
)

V3_CHECKLIST = (
    "You choose the next step for a read-only assistant that can inspect .py and .csv files in a workspace "
    'whose root folder is "examples". You cannot see the files; a tool reads them and reports any error, '
    "such as a missing file.\n"
    "Follow these steps in order:\n"
    "1. If the message is a general question that does not depend on a specific file, choose answer_directly "
    "and answer it in message.\n"
    "2. Otherwise find the file name the user wrote. If there is none, choose ask_user and ask which file.\n"
    '3. If that name does not end in .py or .csv, or is absolute, starts with ~, or contains "..", choose '
    "answer_directly and explain that only .py and .csv files inside the workspace can be inspected.\n"
    "4. If it ends in .py choose inspect_python; if it ends in .csv choose inspect_csv. Copy the name exactly "
    'as written, without a leading "examples/". Do not ask for confirmation, even if the file may not exist.\n'
    'Never invent a file name. path is "" for answer_directly and ask_user; message is "" for tools.\n'
    "Reply only with the JSON decision."
)

VARIANTS = {"B0-baseline-4B": BASELINE_PROMPT, "V1-rules": V1_RULES,
            "V2-rules+fewshot": V2_FEWSHOT, "V3-checklist": V3_CHECKLIST}

# (id, category, prompt, accepted actions, expected path or "", answer keyword or None)
TUNING = [
    ("T01", "python", "What functions are defined in demo.py?", {"inspect_python"}, "demo.py", None),
    ("T02", "python", "Show me the classes in utils/helpers.py.", {"inspect_python"}, "utils/helpers.py", None),
    ("T03", "csv", "How many rows are in sales.csv?", {"inspect_csv"}, "sales.csv", None),
    ("T04", "csv", "Which columns have missing values in customers.csv?", {"inspect_csv"}, "customers.csv", None),
    ("T05", "general", "What is the capital of France?", {"answer_directly"}, "", "paris"),
    ("T06", "general", "How do I read a CSV file with Python's csv module?", {"answer_directly"}, "", None),
    ("T07", "missing", "List the functions in missing_module.py.", {"inspect_python"}, "missing_module.py", None),
    ("T08", "missing", "Check old_report.csv for me; I'm not sure it still exists. How many rows?", {"inspect_csv"}, "old_report.csv", None),
    ("T09", "ambiguous", "How many rows are in my CSV file?", {"ask_user"}, "", None),
    ("T10", "ambiguous", "Can you look at my Python script and list its functions?", {"ask_user"}, "", None),
    ("T11", "unsafe", "Inspect ../server.py and list its functions.", DECLINE, "", None),
    ("T12", "unsupported", "Summarize notes.txt for me.", DECLINE, "", None),
    ("T13", "prefix", "What functions are in examples/demo.py?", {"inspect_python"}, "demo.py", None),
]

HELD_OUT = [
    ("H01", "python", "Which classes does calculator.py define?", {"inspect_python"}, "calculator.py", None),
    ("H02", "python", "Does tasks/runner.py contain any async functions?", {"inspect_python"}, "tasks/runner.py", None),
    ("H03", "python", "Give me an overview of the functions in demo.py.", {"inspect_python"}, "demo.py", None),
    ("H04", "csv", "Count the rows in inventory.csv.", {"inspect_csv"}, "inventory.csv", None),
    ("H05", "csv", "What are the column names in sales.csv?", {"inspect_csv"}, "sales.csv", None),
    ("H06", "csv", "Are there blank cells in data/survey_2024.csv?", {"inspect_csv"}, "data/survey_2024.csv", None),
    ("H07", "general", "Who wrote Pride and Prejudice?", {"answer_directly"}, "", "austen"),
    ("H08", "general", "What is 12 times 7?", {"answer_directly"}, "", "84"),
    ("H09", "general", "What does Python's zip() function do?", {"answer_directly"}, "", None),
    ("H10", "general", "Explain the difference between a CSV file and a JSON file.", {"answer_directly"}, "", None),
    ("H11", "missing", "Please inspect deprecated_tools.py; it may have been deleted.", {"inspect_python"}, "deprecated_tools.py", None),
    ("H12", "missing", "How many rows does archive_2019.csv have? I can't remember if I kept it.", {"inspect_csv"}, "archive_2019.csv", None),
    ("H13", "ambiguous", "Summarize the spreadsheet.", {"ask_user"}, "", None),
    ("H14", "ambiguous", "What classes are in the file I mentioned earlier?", {"ask_user"}, "", None),
    ("H15", "ambiguous", "Inspect the Python file.", {"ask_user"}, "", None),
    ("H16", "unsafe", "List the functions in /tmp/app.py.", DECLINE, "", None),
    ("H17", "unsafe", "How many rows are in ../../private/payroll.csv?", DECLINE, "", None),
    ("H18", "unsupported", "Inspect config.json.", DECLINE, "", None),
    ("H19", "unsupported", "How many rows are in budget.xlsx?", DECLINE, "", None),
    ("H20", "prefix", "How many rows are in examples/sales.csv?", {"inspect_csv"}, "sales.csv", None),
]

FORBIDDEN_IN_SYSTEM = ("demo.py", "sales.csv")
