"""The code agent (Part 3): checks what a paper states about its experiments against the
code and data of its repository (decisions D6, D11).

The repository is fetched as a snapshot of its core files and read statically; nothing in
it is ever run. Settings about the evaluation data go to `data_checks`, everything else
(hyperparameters, architecture sizes, optimizer, seeds, metrics) to `settings_check`.
Claims about the experiments that the planner lists for the claim loop (off by default)
go to `claim_loop`.
"""

from auditor.agents.code.claim_loop import check_claim
from auditor.agents.code.data_checks import DATA_KEYS, check_data_setting
from auditor.agents.code.defects import check_defects
from auditor.agents.code.method_check import check_method
from auditor.agents.code.settings_check import check_setting
from auditor.infra.context import RunContext
from auditor.planner.code_claims import CLAIM_KEY
from auditor.planner.tasks import DEFECTS_KEY, METHOD_KEY
from auditor.schemas import AuditTask, SettingTarget, TaskKind, TaskResult


async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
    """The result of one code task (decision D19 entry point)."""
    if task.kind is not TaskKind.CODE or not isinstance(task.target, SettingTarget):
        raise ValueError(f"{task.task_id} is not a code task")
    if task.target.key in DATA_KEYS:
        return await check_data_setting(task, ctx)
    if task.target.key == METHOD_KEY:
        return await check_method(task, ctx)
    if task.target.key == DEFECTS_KEY:
        return await check_defects(task, ctx)
    if task.target.key == CLAIM_KEY:
        return await check_claim(task, ctx)
    return await check_setting(task, ctx)
