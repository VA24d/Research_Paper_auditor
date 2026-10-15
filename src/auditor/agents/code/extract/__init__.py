"""Settings found statically in a repository snapshot, in the same keys as the paper side
(`auditor.planner.settings` and `auditor.planner.evaluation_setup`).

Covers hyperparameters and architecture sizes (`learning_rate`, `batch_size`, ...), the
optimizer, seeds, metrics (`eval_metric`) and the evaluation data (`data_split`,
`test_fraction`, `validation_fraction`, `dataset`, example and class counts). Nothing in
the repository is imported or run.
"""

from auditor.agents.code.extract.config_files import scan_json, scan_toml, scan_yaml
from auditor.agents.code.extract.hf_card import CardConfig, CardInfo, parse_card
from auditor.agents.code.extract.notebook import scan_notebook
from auditor.agents.code.extract.python_ast import scan_python
from auditor.agents.code.extract.shell import scan_shell
from auditor.agents.code.extract.split_lists import scan_split_lists

__all__ = [
    "CardConfig",
    "CardInfo",
    "parse_card",
    "scan_json",
    "scan_notebook",
    "scan_python",
    "scan_shell",
    "scan_split_lists",
    "scan_toml",
    "scan_yaml",
]
