import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from copilot import CopilotClient, PermissionHandler
from copilot.session_events import ToolExecutionStartData
from copilot.tools import define_tool

from prompts import (
    COORDINATOR_PROMPT,
    PROPONENT_PROMPT,
    REVIEWER_PROMPT,
    SKEPTIC_PROMPT,
    both_reconsideration_prompt,
    coordinator_check_prompt,
    coordinator_repair_prompt,
    final_report_prompt,
    independent_review_prompt,
    reconsider_prompt,
    sequential_first_mover_prompt,
    sequential_second_mover_prompt,
)


MAX_DISAGREEMENT_LOOPS = 3
SEND_TIMEOUT_SECONDS = 600
MAX_DIFF_CHARS = 60000
SUBPROCESS_TIMEOUT_SECONDS = 60

REASONING_EFFORT_CHOICES = ("low", "medium", "high", "xhigh", "max")
DEFAULT_REASONING_EFFORT = "high"


async def run_readonly_command(args, cwd=None):
    """Run a fixed argument list (no shell) and return stdout, raising on failure."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"Command timed out: {' '.join(args)}")

    if proc.returncode != 0:
        message = stderr.decode(errors="replace").strip()
        raise RuntimeError(message or f"Command failed: {' '.join(args)}")

    return stdout.decode(errors="replace")


async def gather_git_diff(working_directory):
    """Return the current branch's diff against its remote base branch."""
    try:
        base_ref = await run_readonly_command(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            cwd=working_directory,
        )
        base = base_ref.strip().replace("refs/remotes/", "")
    except Exception:
        base = "origin/main"

    try:
        diff = await run_readonly_command(
            ["git", "diff", f"{base}...HEAD"],
            cwd=working_directory,
        )
    except Exception as exc:
        return f"(git diff against {base} unavailable: {exc})"

    diff = diff.strip()
    if not diff:
        return f"(no differences from {base})"
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated] ..."
    return diff


class GithubPrParams(BaseModel):
    pr: int = Field(description="Pull request number to inspect")
    action: Literal["view", "diff", "checks"] = Field(
        description=(
            "Read-only action: 'view' for the description and comments, "
            "'diff' for the code changes, 'checks' for CI status"
        ),
    )


async def _github_pr_handler(params: GithubPrParams) -> str:
    args = ["gh", "pr", params.action, str(params.pr)]
    if params.action == "view":
        args.append("--comments")
    return await run_readonly_command(args)


github_pr_tool = define_tool(
    "github_pr",
    description=(
        "Fetch read-only information about a GitHub pull request in the "
        "current repository via the gh CLI. Use it to review a PR referenced "
        "in the question (description, comments, diff, or CI checks)."
    ),
    handler=_github_pr_handler,
    params_type=GithubPrParams,
    skip_permission=True,
)



def summarize_tool_arguments(arguments):
    """Return a short, human-readable summary of a tool call's arguments."""
    if isinstance(arguments, dict):
        interesting = ("path", "file", "pattern", "query", "glob", "command")
        parts = [
            f"{key}={arguments[key]}"
            for key in interesting
            if arguments.get(key)
        ]
        if parts:
            return " ".join(parts)
    return ""


def make_tool_logger(label):
    """Build an event handler that prints each tool the agent invokes."""
    def handler(event):
        if isinstance(event.data, ToolExecutionStartData):
            summary = summarize_tool_arguments(event.data.arguments)
            line = f"[debug] {label} tool: {event.data.tool_name}"
            print(f"{line} {summary}".rstrip())

    return handler




def choose_model(models, preference_groups, family_name):
    for required_terms in preference_groups:
        matches = []

        for model in models:
            searchable = f"{model.id} {model.name}".lower()

            if all(term.lower() in searchable for term in required_terms):
                matches.append(model)

        if matches:
            return min(matches, key=lambda m: len(m.id)).id

    available = "\n".join(
        f"  {model.id}: {model.name}"
        for model in models
    )

    raise RuntimeError(
        f"Could not find an available {family_name} model.\n\n"
        f"Available models:\n{available}"
    )


async def create_reviewer(
    client,
    *,
    name,
    model,
    working_directory,
    max_ai_credits,
    reasoning_effort,
    system_prompt=REVIEWER_PROMPT,
    debug=False,
):
    session = await client.create_session(
        model="auto",
        working_directory=str(working_directory),
        session_limits={
            "max_ai_credits": max_ai_credits,
        },
        tools=[github_pr_tool],
        custom_agents=[
            {
                "name": name,
                "display_name": name,
                "description": "Independent technical reviewer",
                "model": model,
                "reasoning_effort": reasoning_effort,
                "tools": ["grep", "glob", "view", "github_pr"],
                "prompt": system_prompt,
            }
        ],
        agent=name,
        on_permission_request=PermissionHandler.approve_all,
        custom_agents_local_only=True,
        enable_config_discovery=False,
        enable_skills=False,
        enable_session_store=False,
    )

    if debug:
        session.on(make_tool_logger(name))

    return session


async def create_coordinator(
    client,
    *,
    working_directory,
    max_ai_credits,
    reasoning_effort,
    debug=False,
):
    session = await client.create_session(
        model="auto",
        working_directory=str(working_directory),
        session_limits={
            "max_ai_credits": max_ai_credits,
        },
        custom_agents=[
            {
                "name": "coordinator",
                "display_name": "Debate Coordinator",
                "description": "Neutral moderator for AI reviewers",
                "reasoning_effort": reasoning_effort,
                "tools": [],
                "prompt": COORDINATOR_PROMPT,
            }
        ],
        agent="coordinator",
        custom_agents_local_only=True,
        enable_config_discovery=False,
        enable_skills=False,
        enable_session_store=False,
    )

    if debug:
        session.on(make_tool_logger("coordinator"))

    return session


def response_content(response):
    if response is None:
        raise RuntimeError("Copilot returned no response.")

    content = getattr(response.data, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            f"Copilot returned an unexpected {response.type.value} event."
        )

    return content


async def send_prompt(session, prompt, label):
    try:
        response = await session.send_and_wait(
            prompt,
            timeout=SEND_TIMEOUT_SECONDS,
        )
        return response_content(response)
    except Exception as exc:
        raise RuntimeError(f"{label} failed: {exc}") from exc


async def send_concurrently(*requests):
    results = await asyncio.gather(
        *(send_prompt(session, prompt, label) for label, session, prompt in requests),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"Concurrent reviewer call failed:\n{details}")
    return tuple(results)


def _extract_agreements(decision):
    """Leniently read the optional 'agreements' field.

    Accepts a string or a list of strings and never raises, so a missing
    or malformed value simply yields None instead of triggering a repair.
    """
    agreements = decision.get("agreements")
    if isinstance(agreements, list):
        agreements = "; ".join(
            str(item).strip() for item in agreements if str(item).strip()
        )
    if not isinstance(agreements, str) or not agreements.strip():
        return None
    return agreements.strip()


def parse_coordinator_decision(raw):
    raw = raw.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {raw}") from exc

    if not isinstance(decision, dict) or type(decision.get("consensus")) is not bool:
        raise ValueError("decision must contain a boolean 'consensus'")

    if decision["consensus"]:
        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("consensus decision requires a non-empty 'reason'")
        return {
            "consensus": True,
            "reason": reason.strip(),
            "agreements": _extract_agreements(decision),
        }

    strategy = decision.get("strategy")
    focus = decision.get("focus")
    instructions = decision.get("instructions")
    first_reviewer = decision.get("first_reviewer")
    if strategy not in {"BOTH", "SEQUENTIAL"}:
        raise ValueError("strategy must be BOTH or SEQUENTIAL")
    if not isinstance(focus, str) or not focus.strip():
        raise ValueError("non-consensus decision requires a non-empty 'focus'")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("non-consensus decision requires non-empty 'instructions'")
    if strategy == "BOTH" and first_reviewer is not None:
        raise ValueError("first_reviewer must be null for BOTH")
    if strategy == "SEQUENTIAL" and first_reviewer not in {"A", "B"}:
        raise ValueError("first_reviewer must be A or B for SEQUENTIAL")

    return {
        "consensus": False,
        "strategy": strategy,
        "focus": focus.strip(),
        "first_reviewer": first_reviewer,
        "instructions": instructions.strip(),
        "agreements": _extract_agreements(decision),
    }


async def coordinator_decision(
    coordinator,
    *,
    question,
    reviewer_a,
    reviewer_b,
    loop_number,
    min_rounds=0,
    adversarial=False,
):
    raw = await send_prompt(
        coordinator,
        coordinator_check_prompt(
            question,
            reviewer_a,
            reviewer_b,
            loop_number,
            min_rounds,
            adversarial,
        ),
        "Coordinator decision",
    )
    try:
        return parse_coordinator_decision(raw)
    except ValueError as first_error:
        repaired = await send_prompt(
            coordinator,
            coordinator_repair_prompt(first_error),
            "Coordinator decision repair",
        )
        try:
            return parse_coordinator_decision(repaired)
        except ValueError as second_error:
            raise RuntimeError(
                "Coordinator returned malformed decisions twice: "
                f"{second_error}"
            ) from second_error


async def run_both_reconsideration(
    *,
    gpt,
    claude,
    question,
    gpt_position,
    claude_position,
    decision,
):
    focus = decision["focus"]
    instructions = decision["instructions"]

    gpt_position, claude_position = await send_concurrently(
        ("Reviewer A", gpt, both_reconsideration_prompt(question, focus, instructions, claude_position)),
        ("Reviewer B", claude, both_reconsideration_prompt(question, focus, instructions, gpt_position)),
    )

    return gpt_position, claude_position


async def run_sequential_reconsideration(
    *,
    gpt,
    claude,
    question,
    gpt_position,
    claude_position,
    decision,
):
    focus = decision["focus"]
    instructions = decision["instructions"]
    first = decision["first_reviewer"]

    if first not in {"A", "B"}:
        raise RuntimeError(
            "SEQUENTIAL strategy requires first_reviewer A or B."
        )

    # ---------------------------------------------------------
    # Reviewer A goes first
    # ---------------------------------------------------------

    if first == "A":
        new_gpt_position = await send_prompt(
            gpt,
            sequential_first_mover_prompt(question, focus, instructions, claude_position),
            "Reviewer A",
        )
        new_claude_position = await send_prompt(
            claude,
            sequential_second_mover_prompt(question, focus, "A", new_gpt_position, claude_position),
            "Reviewer B",
        )
        return new_gpt_position, new_claude_position

    # ---------------------------------------------------------
    # Reviewer B goes first
    # ---------------------------------------------------------

    new_claude_position = await send_prompt(
        claude,
        sequential_first_mover_prompt(question, focus, instructions, gpt_position),
        "Reviewer B",
    )
    new_gpt_position = await send_prompt(
        gpt,
        sequential_second_mover_prompt(question, focus, "B", new_claude_position, gpt_position),
        "Reviewer A",
    )

    return new_gpt_position, new_claude_position


async def create_final_report(
    coordinator,
    *,
    question,
    gpt_position,
    claude_position,
    consensus_reached,
    loops_used,
    adversarial=False,
):
    status = (
        "The reviewers reached material consensus."
        if consensus_reached
        else (
            "The maximum disagreement rounds were reached. "
            "Some disagreement may remain."
        )
    )

    return await send_prompt(
        coordinator,
        final_report_prompt(question, gpt_position, claude_position, status, loops_used, adversarial),
        "Final report",
    )


def print_round_debug(label, decision):
    agreements = decision.get("agreements") or "none stated"
    print(f"[debug] {label} agreements: {agreements}")
    if decision["consensus"]:
        print(f"[debug] {label} disagreement: none (consensus)")
    else:
        print(f"[debug] {label} disagreement: {decision['focus']}")


async def run_debate(
    question,
    max_ai_credits,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    min_rounds=0,
    adversarial=False,
    debug=False,
):
    working_directory = Path.cwd()

    client = CopilotClient()
    await client.start()

    try:
        models = await client.list_models()

        gpt_model = choose_model(
            models,
            [
                ["gpt", "5.6", "sol"],
                ["gpt", "5.4"],
            ],
            "GPT-5.6 Sol",
        )

        claude_model = choose_model(
            models,
            [
                ["claude", "opus", "5"],
                ["claude", "opus", "4.8", "fast"],
                ["claude", "opus", "4.8"],
                ["claude", "opus", "4.7"],
                ["claude", "opus"],
            ],
            "Claude Opus",
        )

        print()
        print(f"GPT reviewer:       {gpt_model}")
        print(f"Claude reviewer:    {claude_model}")
        print(f"Working directory:  {working_directory}")
        print(f"Credit cap/session: {max_ai_credits}")
        print(f"Reasoning effort:   {reasoning_effort}")
        print(f"Min disagreement rounds: {min_rounds}")
        print(
            f"Max disagreement loops: "
            f"{MAX_DISAGREEMENT_LOOPS}"
        )
        if adversarial:
            print("Mode: adversarial (A=proponent, B=skeptic)")
        print()

        gpt_prompt = PROPONENT_PROMPT if adversarial else REVIEWER_PROMPT
        claude_prompt = SKEPTIC_PROMPT if adversarial else REVIEWER_PROMPT

        gpt = await create_reviewer(
            client,
            name="reviewer-gpt",
            model=gpt_model,
            working_directory=working_directory,
            max_ai_credits=max_ai_credits,
            reasoning_effort=reasoning_effort,
            system_prompt=gpt_prompt,
            debug=debug,
        )

        try:
            claude = await create_reviewer(
                client,
                name="reviewer-claude",
                model=claude_model,
                working_directory=working_directory,
                max_ai_credits=max_ai_credits,
                reasoning_effort=reasoning_effort,
                system_prompt=claude_prompt,
                debug=debug,
            )
        except Exception:
            await gpt.disconnect()
            raise

        try:
            coordinator = await create_coordinator(
                client,
                working_directory=working_directory,
                max_ai_credits=max_ai_credits,
                reasoning_effort=reasoning_effort,
                debug=debug,
            )
        except Exception:
            await asyncio.gather(
                gpt.disconnect(),
                claude.disconnect(),
                return_exceptions=True,
            )
            raise

        try:
            # ============================================================
            # ROUND 0 — Independent reviews
            # ============================================================

            print("=== Round 0: independent reviews ===")

            git_diff = await gather_git_diff(working_directory)
            if debug:
                print(f"[debug] injected git diff: {len(git_diff)} chars")

            reviewer_question = independent_review_prompt(question, git_diff)
            gpt_position, claude_position = await send_concurrently(
                ("Reviewer A", gpt, reviewer_question),
                ("Reviewer B", claude, reviewer_question),
            )

            # ============================================================
            # ROUND 1 — Mandatory initial reconsideration
            # ============================================================

            print("=== Round 1: initial reconsideration ===")

            gpt_position, claude_position = await send_concurrently(
                ("Reviewer A", gpt, reconsider_prompt(question, gpt_position, claude_position)),
                ("Reviewer B", claude, reconsider_prompt(question, claude_position, gpt_position)),
            )

            # ============================================================
            # Coordinator checks + up to N disagreement loops
            # ============================================================

            consensus_reached = False
            loops_used = 0

            for loop_number in range(
                1,
                MAX_DISAGREEMENT_LOOPS + 1,
            ):
                print(
                    f"=== Coordinator check before "
                    f"disagreement loop {loop_number} ==="
                )

                decision = await coordinator_decision(
                    coordinator,
                    question=question,
                    reviewer_a=gpt_position,
                    reviewer_b=claude_position,
                    loop_number=loop_number,
                    min_rounds=min_rounds,
                    adversarial=adversarial,
                )

                if debug:
                    print_round_debug(f"Round {loop_number}", decision)

                if decision["consensus"]:
                    consensus_reached = True

                    print("Coordinator: consensus reached.")
                    print(
                        "Reason:",
                        decision.get("reason", "No reason provided."),
                    )

                    break

                loops_used += 1

                strategy = decision["strategy"]
                focus = decision["focus"]

                print("Coordinator: disagreement remains.")
                print(f"Focus: {focus}")
                print(f"Strategy: {strategy}")

                # --------------------------------------------------------
                # Strategy 1: BOTH reconsider concurrently
                # --------------------------------------------------------

                if strategy == "BOTH":
                    gpt_position, claude_position = (
                        await run_both_reconsideration(
                            gpt=gpt,
                            claude=claude,
                            question=question,
                            gpt_position=gpt_position,
                            claude_position=claude_position,
                            decision=decision,
                        )
                    )

                # --------------------------------------------------------
                # Strategy 2: sequential reconsideration
                # --------------------------------------------------------

                elif strategy == "SEQUENTIAL":
                    print(
                        "First reviewer:",
                        decision["first_reviewer"],
                    )

                    gpt_position, claude_position = (
                        await run_sequential_reconsideration(
                            gpt=gpt,
                            claude=claude,
                            question=question,
                            gpt_position=gpt_position,
                            claude_position=claude_position,
                            decision=decision,
                        )
                    )

                else:
                    raise RuntimeError(
                        f"Unknown coordinator strategy: {strategy}"
                    )

            # ============================================================
            # Final consensus check
            #
            # Important:
            # If loop 3 itself resolved the disagreement, we should notice
            # that rather than incorrectly saying we stopped unresolved.
            # ============================================================

            if not consensus_reached:
                print("=== Final coordinator consensus check ===")

                final_decision = await coordinator_decision(
                    coordinator,
                    question=question,
                    reviewer_a=gpt_position,
                    reviewer_b=claude_position,
                    loop_number=MAX_DISAGREEMENT_LOOPS + 1,
                    adversarial=adversarial,
                )

                if debug:
                    print_round_debug("Final check", final_decision)

                consensus_reached = final_decision["consensus"]

            # ============================================================
            # Final report
            # ============================================================

            print()
            print("=== Final report ===")
            print()

            final_report = await create_final_report(
                coordinator,
                question=question,
                gpt_position=gpt_position,
                claude_position=claude_position,
                consensus_reached=consensus_reached,
                loops_used=loops_used,
                adversarial=adversarial,
            )

            print(f"Reviewer A: {gpt_model}")
            print(f"Reviewer B: {claude_model}")
            print(
                f"Consensus: "
                f"{'YES' if consensus_reached else 'NO'}"
            )
            print(f"Disagreement loops: {loops_used}")
            print()
            print(final_report)

        finally:
            await asyncio.gather(
                gpt.disconnect(),
                claude.disconnect(),
                coordinator.disconnect(),
                return_exceptions=True,
            )

    finally:
        await client.stop()


async def list_models():
    client = CopilotClient()
    await client.start()

    try:
        models = await client.list_models()

        for model in models:
            print(f"{model.id:35} {model.name}")

    finally:
        await client.stop()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Moderated multi-model technical discussion "
            "using GitHub Copilot SDK."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Question for the reviewers",
    )

    parser.add_argument(
        "--credits",
        type=float,
        default=600,
        help=(
            "Soft AI credit limit per Copilot session "
            "(default: 600)"
        ),
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Show models available to your Copilot account",
    )

    parser.add_argument(
        "--effort",
        choices=REASONING_EFFORT_CHOICES,
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Model reasoning (thinking) effort for reviewers and "
            f"coordinator (default: {DEFAULT_REASONING_EFFORT})"
        ),
    )

    parser.add_argument(
        "--min-rounds",
        type=int,
        default=0,
        help=(
            "Force at least this many disagreement rounds before the "
            "coordinator may declare consensus (default: 0, max: "
            f"{MAX_DISAGREEMENT_LOOPS})"
        ),
    )

    parser.add_argument(
        "--adversarial",
        action="store_true",
        help=(
            "Assign opposing sides: Reviewer A argues for the change "
            "(proponent) and Reviewer B against it (skeptic), to force a "
            "genuine multi-round debate"
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print the coordinator's agreements and disagreement "
            "for each round, plus each tool the agents invoke"
        ),
    )

    args = parser.parse_args()

    if args.list_models:
        asyncio.run(list_models())
        return

    if args.credits < 30:
        parser.error("--credits must be at least 30 (Copilot SDK minimum)")

    if not 0 <= args.min_rounds <= MAX_DISAGREEMENT_LOOPS:
        parser.error(
            f"--min-rounds must be between 0 and {MAX_DISAGREEMENT_LOOPS}"
        )

    question = (args.question or input("Question: ")).strip()
    if not question:
        parser.error("a non-empty question is required")

    asyncio.run(
        run_debate(
            question=question,
            max_ai_credits=args.credits,
            reasoning_effort=args.effort,
            min_rounds=args.min_rounds,
            adversarial=args.adversarial,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()