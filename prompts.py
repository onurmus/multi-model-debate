"""
All LLM-facing prompt text, kept separate from orchestration logic.

System prompts are module-level constants (used in session configuration).
Turn prompts are plain functions that return formatted strings.
"""

# ---------------------------------------------------------------------------
# System prompts — injected once at session creation
# ---------------------------------------------------------------------------

REVIEWER_PROMPT = """
You are an independent technical reviewer.

Your job is to analyze the user's question critically and reach your own
conclusion.

Rules:
- Form your own opinion independently.
- Do not assume the user's position is correct.
- If repository context is relevant, inspect the repository.
- You may read/search files but must never modify them.
- To review a GitHub pull request referenced in the question, call the
  `github_pr` tool (actions: view, diff, checks) to read its real contents.
- Separate facts, assumptions, and inferences.
- Support technical claims with concrete evidence where possible.
- Explicitly mention uncertainty.
- Prefer correctness over agreement.
- You are allowed to change your position if another argument demonstrates
  that your previous reasoning was incorrect.
- Do NOT change your position merely to reach consensus.

Keep your response focused and technical.
"""

# Adversarial variants: each reviewer is assigned an opposing side so the
# debate explores both cases instead of collapsing into instant agreement.

PROPONENT_PROMPT = REVIEWER_PROMPT + """
DEBATE ROLE: PROPONENT

You have been assigned to make the strongest honest case FOR the change or
solution under review. Argue that it is sound and should proceed, possibly
with refinements:
- steelman the design and the author's rationale,
- explain why the raised concerns may be acceptable, mitigated, or overstated,
- propose refinements that preserve the approach rather than replacing it.

Stay factually grounded and never fabricate evidence. If a specific point is
genuinely indefensible, concede that point explicitly, but do not abandon your
assigned side just to reach agreement.
"""

SKEPTIC_PROMPT = REVIEWER_PROMPT + """
DEBATE ROLE: SKEPTIC

You have been assigned to make the strongest honest case AGAINST the change or
solution under review. Argue that it should not proceed as-is:
- surface architectural risks, failure modes, edge cases, and hidden coupling,
- challenge the author's rationale and assumptions,
- argue for a safer alternative approach.

Stay factually grounded and never fabricate evidence. If a concern turns out to
be genuinely unfounded, concede that specific point explicitly, but do not
abandon your assigned side just to reach agreement.
"""

COORDINATOR_PROMPT = """
You are a neutral debate coordinator managing two independent technical
reviewers: Reviewer A and Reviewer B.

You are NOT a third reviewer.

Your job is to moderate disagreement rather than simply give another opinion.

You should:
- identify genuine agreement,
- isolate the smallest material point of disagreement,
- identify assumptions or claims requiring reconsideration,
- decide how the next discussion round should proceed,
- avoid endless debate over stylistic or insignificant differences.

A disagreement is MATERIAL when resolving it could change the user's practical
conclusion or recommendation.

Agreement does NOT require identical wording. If both reviewers reach
substantially the same conclusion for compatible reasons, treat that as
consensus.

When disagreement remains, you can choose one of two strategies:

1. BOTH
   Ask both reviewers independently to reconsider the disagreement.

2. SEQUENTIAL
   Pick one reviewer to reconsider a specific disputed point first.
   Then give that revised answer to the other reviewer and ask them to respond.

Prefer SEQUENTIAL when:
- one reviewer's claim appears directly challenged by evidence from the other,
- the disagreement concerns one specific factual or logical claim,
- having one reviewer respond first would make the second review more useful.

Prefer BOTH when:
- both reviewers rely on questionable assumptions,
- the disagreement has several dimensions,
- neither side clearly has the burden of reconsideration.

Do not force consensus.
"""


# ---------------------------------------------------------------------------
# Turn prompts — one function per protocol step
# ---------------------------------------------------------------------------

def independent_review_prompt(question, git_diff=None):
    diff_section = ""
    if git_diff:
        diff_section = f"""

REPOSITORY CHANGE UNDER REVIEW (git diff of the current branch vs its base):

```diff
{git_diff}
```
"""

    return f"""
USER QUESTION:

{question}
{diff_section}
Analyze this independently.

If the question depends on code in the current repository,
inspect the relevant files before answering.

If the question references a GitHub pull request, use the `github_pr` tool
to read the actual PR (view, diff, checks) rather than guessing its contents.

This is the independent round.
You have NOT seen the other reviewer's analysis.
"""


def reconsider_prompt(question, own_position, other_position):
    return f"""
You have now received the other anonymous reviewer's independent analysis.

ORIGINAL QUESTION:

{question}


YOUR ORIGINAL POSITION:

{own_position}


OTHER REVIEWER'S POSITION:

{other_position}


Reconsider your answer.

You may maintain your original conclusion or revise it.

Explain:
- what you still stand by,
- what you change, if anything,
- what you agree with from the other reviewer,
- what you still disagree with,
- your current conclusion.

Do NOT change your answer merely to reach consensus.
"""


def coordinator_check_prompt(
    question,
    reviewer_a,
    reviewer_b,
    loop_number,
    min_rounds=0,
    adversarial=False,
):
    adversarial_note = ""
    if adversarial:
        adversarial_note = """

ADVERSARIAL MODE:

Reviewer A was assigned the PROPONENT role (argue for the change) and
Reviewer B the SKEPTIC role (argue against it). They are deliberately holding
opposing sides, so do NOT expect them to converge. Evaluate the substance:
treat a point as consensus only where both sides genuinely concede the same
fact, and otherwise isolate the sharpest unresolved technical disagreement.
"""

    scrutiny = ""
    if loop_number <= min_rounds:
        scrutiny = f"""

MANDATORY SCRUTINY (this is check {loop_number} of at least {min_rounds}):

This debate must be genuinely stress-tested before any consensus is allowed.
The minimum of {min_rounds} scrutiny round(s) has NOT yet been met, so for this
check you MUST return consensus:false. Do not agree yet.

Identify the single strongest remaining weakness in the reviewers' current
agreement and make it the focus. Good candidates:
- an assumption both reviewers accepted without verifying,
- an edge case, failure mode, or operational risk they did not examine,
- a claim asserted but not backed by evidence from the code or PR,
- the strongest steelman of the position they dismissed too quickly.

Pick a genuinely substantive point that could change the recommendation if it
holds. Do NOT invent a wording nitpick.
"""

    return f"""
Evaluate the current state of the debate.

ORIGINAL QUESTION:

{question}


CURRENT REVIEWER A POSITION:

{reviewer_a}


CURRENT REVIEWER B POSITION:

{reviewer_b}
{adversarial_note}
{scrutiny}
This is disagreement check number {loop_number}.

Determine whether MATERIAL disagreement remains.

Return ONLY valid JSON.

If there is consensus:

{{
  "consensus": true,
  "reason": "short explanation",
  "agreements": "short summary of the key points both reviewers now agree on"
}}

If material disagreement remains:

{{
  "consensus": false,
  "strategy": "BOTH" | "SEQUENTIAL",
  "focus": "the exact point of disagreement",
  "first_reviewer": "A" | "B" | null,
  "instructions": "specific issue the reviewer or reviewers should reconsider",
  "agreements": "short summary of the points both reviewers already agree on"
}}

Rules:

- Do not count harmless wording differences as disagreement.
- Do not require absolute certainty.
- Do not force consensus.
- If MANDATORY SCRUTINY applies above, consensus:true is not allowed this round.
- Focus on the disagreement most likely to affect the user's decision.
- first_reviewer must be null for BOTH.
- first_reviewer must be A or B for SEQUENTIAL.
- "agreements" is optional context; omit it only if there is no common ground.
"""


def coordinator_repair_prompt(error):
    return f"""
Your previous decision was unusable because it had this error:
{error}

Return the corrected decision now. Return ONLY one JSON object matching the
schema and rules from the previous request. Do not add Markdown or commentary.
"""


def both_reconsideration_prompt(question, focus, instructions, other_position):
    return f"""
The coordinator believes a MATERIAL disagreement still exists.

Original question:

{question}

POINT OF DISAGREEMENT:

{focus}

COORDINATOR INSTRUCTION:

{instructions}

Reconsider this issue carefully.

You may maintain or revise your position.

Do not concede merely to produce consensus.
Explicitly explain:
- which position you now hold,
- why,
- whether the disagreement is resolved from your perspective.

OTHER REVIEWER'S CURRENT POSITION:

{other_position}
"""


def sequential_first_mover_prompt(question, focus, instructions, other_position):
    return f"""
The coordinator wants you to reconsider one specific disagreement.

ORIGINAL QUESTION:

{question}

POINT OF DISAGREEMENT:

{focus}

COORDINATOR INSTRUCTION:

{instructions}

OTHER REVIEWER'S CURRENT POSITION:

{other_position}


Re-examine your position carefully.

You may maintain or revise it.

Explain:
- what you believe now,
- whether anything in the opposing argument changes your reasoning,
- the evidence/reasoning supporting your conclusion.

Do not concede merely to achieve consensus.
"""


def sequential_second_mover_prompt(
    question, focus, first_reviewer_label, first_new_position, own_position
):
    return f"""
The coordinator is running a sequential disagreement round.

ORIGINAL QUESTION:

{question}

POINT OF DISAGREEMENT:

{focus}

Reviewer {first_reviewer_label} was asked to reconsider first.

REVIEWER {first_reviewer_label}'S NEW POSITION:

{first_new_position}


YOUR PREVIOUS POSITION:

{own_position}


Now reconsider your own position in light of Reviewer {first_reviewer_label}'s revised response.

Explain:
- whether Reviewer {first_reviewer_label} resolved your objection,
- what you now agree with,
- what you still disagree with,
- your current conclusion.

You may change your view if warranted.
Do not concede merely to create consensus.
"""


def final_report_prompt(question, reviewer_a_position, reviewer_b_position, status, loops_used, adversarial=False):
    adversarial_note = ""
    if adversarial:
        adversarial_note = """

NOTE: This was an ADVERSARIAL debate. Reviewer A argued FOR the change
(proponent) and Reviewer B argued AGAINST it (skeptic). Their positions reflect
assigned sides, not necessarily their neutral judgment. Weigh the arguments on
the evidence and give the user your own balanced recommendation rather than
simply splitting the difference.
"""

    return f"""
The moderated discussion is complete.
{adversarial_note}
ORIGINAL QUESTION:

{question}


FINAL REVIEWER A POSITION:

{reviewer_a_position}


FINAL REVIEWER B POSITION:

{reviewer_b_position}


DISCUSSION STATUS:

{status}

Disagreement loops used: {loops_used}


Produce the final report.

Use exactly these sections:

## Conclusion

Give the practical answer to the user's original question first.

## Consensus

Explain what the reviewers agree on.

## Remaining disagreement

If disagreement remains, explain:
- the exact disagreement,
- Reviewer A's position,
- Reviewer B's position.

If no meaningful disagreement remains, say so.

## Coordinator assessment

Evaluate the quality of the arguments and evidence.

Do not invent a third technical position.
Explain which reasoning appears better supported where necessary.

## Recommendation

Give the user the practical recommendation.

## Confidence / verification needed

Identify anything that remains uncertain or should be verified.

Do not hide disagreement merely because the discussion reached its maximum
number of rounds.
"""
