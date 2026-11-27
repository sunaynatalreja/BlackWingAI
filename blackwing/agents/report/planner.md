# Report task planner

Break the report into the four fixed tasks in the required order. Return only
the four task IDs, each exactly once:

1. `overview` — summarize the reported event and identify what is established.
2. `findings_and_causes` — separate supported observations from hypotheses and
   identify what can and cannot be said about cause.
3. `comparisons_and_compliance` — summarize historical analogies and
   regulatory/procedural assessment without transferring findings between cases.
4. `risk_and_follow_up` — summarize candidate hazards, proposed actions, open
   questions, verification gaps, and human-review needs.

Return exactly this JSON shape, using the `task_ids` key (never `task_order`):

{"task_ids":["overview","findings_and_causes","comparisons_and_compliance","risk_and_follow_up"]}

The plan is a work order only. Do not include incident facts or free-form
instructions. The available task IDs are constrained by the output schema.
