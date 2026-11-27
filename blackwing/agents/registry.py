"""Central workflow registration; agent owners edit their own package, not this list's prompts."""

from blackwing.agents.base import AgentSpec
from blackwing.agents.investigation import SPEC as INVESTIGATION
from blackwing.agents.similar_case import SPEC as SIMILAR_CASE
from blackwing.agents.compliance import SPEC as COMPLIANCE
from blackwing.agents.risk import SPEC as RISK
from blackwing.agents.mitigation import SPEC as MITIGATION
from blackwing.agents.critic import SPEC as CRITIC
from blackwing.agents.verification import SPEC as VERIFICATION
from blackwing.agents.report import SPEC as REPORT


AGENT_SPECS: tuple[AgentSpec, ...] = (
    INVESTIGATION,
    SIMILAR_CASE,
    COMPLIANCE,
    RISK,
    MITIGATION,
    CRITIC,
    VERIFICATION,
    REPORT,
)
AGENT_BY_ID = {spec.key: spec for spec in AGENT_SPECS}
