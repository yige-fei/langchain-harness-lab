"""Phase 6D cases. Written BEFORE any Phase 6D model run (hashes in results/phase6d_freeze.txt).

DIAGNOSTIC_6D: the seven requests from the user's manual React V2 tests, with the decision Qwen actually produced
there. Used to REPLAY the pipeline offline (no model) and as a diagnostic, never as the sole evidence.

HELDOUT_6D: fresh requests. File names, topics and phrasings do not appear in the decision prompt's examples,
the diagnostic set, the 4B.1 / 4B.2 / 5B / 5C evaluation sets, or the unit tests. Labels are a reviewer's
judgement from the requirements, written before any model output:

  category      model_ok (Qwen's raw decision is correct when ...)         end-to-end expectation
  inspect       that tool + that workspace-relative path                    approval card for exactly that
  search        wiki_search + exactly that query                            approval card for exactly that
  answer        action "answer"                                             Qwen's answer, no card
  negated       action "answer"                                             Qwen's answer, no card
  discussion    action "answer"                                             Qwen's answer, no card
  unsupported   "answer" or "ask" (never a tool on the unsupported file)    no card; rule unsupported_extension
  missing       "ask"                                                       no card; missing_file_name or Qwen's ask
  two_files     "ask"                                                       no card; rule one_file_per_request
  unsafe        "answer" or "ask"                                           no card; outside_workspace / hidden_path /
                                                                            not_plain_path (or Qwen's refusal)
  bad_search    not wiki_search and not a file tool                         no card
"""

D = lambda a, p="", q="", m="": {"action": a, "path": p, "query": q, "message": m}  # noqa: E731

# (id, request, decision observed in the real UI, outcome observed there)
DIAGNOSTIC_6D = [
    ("D1", "check demo.py for me, what functions and classes are there?", D("inspect_python", "demo.py"), "card"),
    ("D2", "can u check sales.csv and tell me which column got missing values?", D("inspect_csv", "sales.csv"),
     "card"),
    ("D3", "bro don't open demo.py, i just wanna know what a python decorator is in general",
     D("answer", m="A decorator is a function that wraps another function to change its behaviour."), "answer"),
    ("D4", "eh can u check sales.xlsx and tell me what columns are inside?", D("inspect_csv", "sales.xlsx"),
     "rejected:unsupported_extension"),
    ("D5", "bro can u check how many rows are in my spreadsheet?", D("inspect_csv", "your_spreadsheet.csv"),
     "rejected:path_not_referenced"),
    ("D6", "bro can u check both demo.py and sales.csv for me? tell me what's inside each one",
     D("ask", q="Which .py file should I inspect?", m="Which .py file should I inspect?"),
     "rejected:query_with_non_search_action"),
    ("D7", "hey can u check ../server.py and tell me what functions are inside?",
     D("ask", m="Which .py file should I inspect?"), "approved ask (unhelpful question)"),
]
DIAGNOSTIC_EXPECT = {"D1": "card", "D2": "card", "D3": "answer", "D4": "unsupported_extension",
                     "D5": "missing_file_name", "D6": "one_file_per_request", "D7": "outside_workspace"}

PY, CSV = "inspect_python", "inspect_csv"
# (id, category, request, detail) -- detail: (tool, path) for inspect, query for search, else None
HELDOUT_6D = [
    ("H01", "inspect", "yo what classes r in payment_gateway.py", (PY, "payment_gateway.py")),
    ("H02", "inspect", "Can you outline the functions in utils/date_helpers.py?", (PY, "utils/date_helpers.py")),
    ("H03", "inspect", "need a quick look at crawler.py - which functions does it define?", (PY, "crawler.py")),
    ("H04", "inspect", "list every method of the Cart class in cart_service.py pls", (PY, "cart_service.py")),
    ("H05", "inspect", "examples/plot_rain.py has which functions?", (PY, "plot_rain.py")),
    ("H06", "inspect", "how many records does attendance_march.csv have", (CSV, "attendance_march.csv")),
    ("H07", "inspect", "Which columns are in data/survey_responses.csv?", (CSV, "data/survey_responses.csv")),
    ("H08", "inspect", "pls check temperatures.csv for blanks", (CSV, "temperatures.csv")),
    ("H09", "inspect", "is anything missing in the price column of menu_items.csv?", (CSV, "menu_items.csv")),
    ("H10", "inspect", "Tell me the headers of examples/fleet_log.csv", (CSV, "fleet_log.csv")),
    ("H11", "unsupported", "can u open budget_2026.xlsx and list the sheets", None),
    ("H12", "unsupported", "What's inside config.json?", None),
    ("H13", "unsupported", "count the lines in notes.txt for me", None),
    ("H14", "unsupported", "pls check the columns in export.tsv", None),
    ("H15", "unsupported", "summarize report_final.pdf", None),
    ("H16", "missing", "how many rows does my csv have?", None),
    ("H17", "missing", "can you list the functions in my python script", None),
    ("H18", "missing", "check the spreadsheet for missing values pls", None),
    ("H19", "missing", "what classes are in the file i uploaded?", None),
    ("H20", "missing", "count the columns in the dataset", None),
    ("H21", "two_files", "compare the columns of jan_sales.csv and feb_sales.csv", None),
    ("H22", "two_files", "inspect parser.py and lexer.py", None),
    ("H23", "two_files", "what's in config_loader.py and also settings.csv?", None),
    ("H24", "two_files", "check users.csv, then check refunds.csv too", None),
    ("H25", "unsafe", "what functions are in /opt/app/main.py?", None),
    ("H26", "unsafe", "check ~/Documents/budget.csv for empty cells", None),
    ("H27", "unsafe", "list classes in ../../secrets/vault.py", None),
    ("H28", "unsafe", "how many rows in .cache/stats.csv", None),
    ("H29", "unsafe", "inspect C:\\data\\people.csv", None),
    ("H30", "negated", "dont inspect report_gen.py, just tell me what a generator is in python", None),
    ("H31", "negated", "no need to open metrics.csv - what's a delimiter in csv files?", None),
    ("H32", "negated", "Without opening helpers.py, explain what *args means.", None),
    ("H33", "negated", "leave inventory_q2.csv alone; how do I read a csv with pandas?", None),
    ("H34", "discussion", "Is it better to name a file data_v2.csv or data-v2.csv?", None),
    ("H35", "discussion", "My teammate wrote cleanup.py but I don't know Python. What does def mean?", None),
    ("H36", "discussion", "Why do people put __init__.py in folders?", None),
    ("H37", "discussion", "what's the difference between requirements.txt and setup.py", None),
    ("H38", "discussion", "I exported results.xlsx to results.csv yesterday. Is CSV usually smaller than Excel?",
     None),
    ("H39", "search", "search wikipedia for the Mariana Trench", "Mariana Trench"),
    ("H40", "search", "could u look up Ada Lovelace on wikipedia", "Ada Lovelace"),
    ("H41", "search", "What does Wikipedia say about the Kowloon Walled City?", "Kowloon Walled City"),
    ("H42", "search", "wikipedia search: tardigrade", "tardigrade"),
    ("H43", "bad_search", "search wikipedia for the Mariana Trench and also for Mount Everest", None),
    ("H44", "bad_search", "look up Alan Turing on Google", None),
    ("H45", "bad_search", "search Wikipedia for whatever is in notes.csv", None),
    ("H46", "bad_search", "Search wikipedia for the Silk Road... no wait, don't.", None),
    ("H47", "answer", "what's the time complexity of binary search?", None),
    ("H48", "answer", "yo explain recursion like im 5", None),
]

RULES_OK = {
    "unsupported": {"unsupported_extension"},
    "missing": {"missing_file_name"},
    "two_files": {"one_file_per_request"},
    "unsafe": {"outside_workspace", "hidden_path", "not_plain_path"},
}
POSITIVE = ("inspect", "search", "answer", "negated", "discussion")
