# Multi-model debate

An MVP using the GitHub Copilot SDK to run two isolated technical reviewers and
a neutral coordinator. Reviewer A uses the best available GPT-5.6 Sol match;
Reviewer B uses the best available Claude Opus match. Reviewers can inspect the
current repository with read-only tools.

## Setup

Python 3.11 or newer and a GitHub account with Copilot access are required.
The SDK uses the authenticated Copilot CLI user.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m copilot download-runtime
```

Authenticate with the GitHub Copilot CLI if it is not already signed in, then
confirm which models your account exposes:

```bash
copilot login
python debate.py --list-models
```

## Usage

```bash
python debate.py "Is this implementation thread-safe?"
python debate.py --credits 50 "Review the current architecture."
python debate.py
```

```bash
cd ../GIT/airbyte-k8s-operator
python debate.py "Plesea review the current repository and branch?"
```


`--credits` is a limit for each of the three separate SDK sessions, not a
global debate-wide cap. The SDK currently requires at least 30 credits per
session.

`--effort` sets the model reasoning (thinking) effort for the reviewers and
coordinator. Choices are `low`, `medium`, `high`, `xhigh`, `max`
(default: `high`). Higher effort makes reviewers investigate more deeply
before concluding.

`--min-rounds` forces at least this many disagreement rounds before the
coordinator is allowed to declare consensus (default: `0`, max: `3`). On
clear-cut questions the reviewers often agree immediately; setting e.g.
`--min-rounds 2` makes the coordinator surface the strongest remaining
assumption, edge case, or unverified claim each round and drive genuine
reconsideration before consensus.

`--adversarial` assigns opposing sides: Reviewer A argues FOR the change
(proponent) and Reviewer B argues AGAINST it (skeptic). This produces an
organic multi-round debate that explores both cases instead of collapsing
into instant agreement. Each reviewer still stays factually grounded and
concedes genuinely indefensible points; the final report weighs both sides
and gives a balanced recommendation. Combine with `--min-rounds` for even
deeper exploration.

`--debug` prints the coordinator's agreements and disagreement each round,
and logs every tool the agents invoke (e.g. which files they read), so you
can confirm the reviewers are actually inspecting the repository.

The reviewers read files in the current working directory using `grep`,
`glob`, and `view`. In addition:

- The `git diff` of the current branch against its remote base (`origin/HEAD`,
  typically `origin/main`) is computed automatically and injected into the
  first review round, so the reviewers see exactly what changed.
- A read-only `github_pr` tool lets the reviewers fetch a referenced pull
  request over the network via the `gh` CLI (actions: `view`, `diff`,
  `checks`). This requires `gh` to be installed and authenticated
  (`gh auth login`).

Check out the branch under review before running.

The coordinator explicitly selects concurrent or sequential reconsideration.
At most three disagreement loops run after the mandatory initial
reconsideration, followed by a final consensus check and report.

## Tests

```bash
python -m unittest -v
```