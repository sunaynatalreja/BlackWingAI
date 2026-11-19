# Similar Case Analysis — deterministic comparison

Implemented in `agent.py` without an LLM call. Only compare separately
identified historical incident reports. Source-deduplicate, provide real source
IDs and titles, describe actual shared or unknown characteristics. Never copy a
historical probable cause into current-case findings. If no historical record
was retrieved, report no match rather than invent one.
