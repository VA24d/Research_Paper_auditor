"""Settings in the code cells of Jupyter notebooks, read like Python files.

Evidence must point at a line of the file as stored, so each setting's line within its
cell is mapped back to the line of the notebook's JSON that holds that source line.
IPython magics and shell escapes (`%pip`, `!ls`) are blanked before parsing.
"""

import json

from auditor.agents.code.extract.python_ast import scan_python
from auditor.schemas import ExtractedSetting, RepositoryRef, RepositorySource


def scan_notebook(path: str, text: str, repository: RepositoryRef) -> list[ExtractedSetting]:
    try:
        notebook = json.loads(text)
    except (ValueError, RecursionError):
        return []
    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        return []
    raw_lines = text.splitlines()
    position = 0
    found = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        lines = source.splitlines(keepends=True) if isinstance(source, str) else list(source)
        file_lines = []
        for line in lines:
            encoded = json.dumps(str(line))[1:-1]
            index = next(
                (i for i in range(position, len(raw_lines)) if encoded in raw_lines[i]), None
            )
            file_lines.append(index + 1 if index is not None else None)
            if index is not None:
                position = index
        code = "".join(
            "\n" if str(line).lstrip().startswith(("%", "!")) else str(line) for line in lines
        )
        for setting in scan_python(path, code, repository):
            source_ref = setting.source
            assert isinstance(source_ref, RepositorySource)
            line = file_lines[source_ref.line - 1] if source_ref.line <= len(file_lines) else None
            if line is None:
                continue
            found.append(
                setting.model_copy(update={"source": source_ref.model_copy(update={"line": line})})
            )
    return found
