import unittest
from types import SimpleNamespace

from debate import choose_model, parse_coordinator_decision


class ModelSelectionTests(unittest.TestCase):
    def test_prefers_first_matching_group(self):
        models = [
            SimpleNamespace(id="claude-opus-4.8", name="Claude Opus 4.8"),
            SimpleNamespace(id="claude-opus-5", name="Claude Opus 5"),
        ]

        selected = choose_model(
            models,
            [["claude", "opus", "5"], ["claude", "opus"]],
            "Claude Opus",
        )

        self.assertEqual(selected, "claude-opus-5")

    def test_missing_family_lists_available_models(self):
        models = [SimpleNamespace(id="gpt-5.6-sol", name="GPT-5.6 Sol")]

        with self.assertRaisesRegex(RuntimeError, "gpt-5.6-sol"):
            choose_model(models, [["claude", "opus"]], "Claude Opus")


class CoordinatorDecisionTests(unittest.TestCase):
    def test_accepts_fenced_sequential_decision(self):
        decision = parse_coordinator_decision(
            """```json
{"consensus": false, "strategy": "SEQUENTIAL", "focus": "claim", "first_reviewer": "B", "instructions": "verify claim"}
```"""
        )

        self.assertEqual(decision["first_reviewer"], "B")

    def test_rejects_invalid_both_first_reviewer(self):
        with self.assertRaisesRegex(ValueError, "must be null"):
            parse_coordinator_decision(
                '{"consensus": false, "strategy": "BOTH", "focus": "claim", '
                '"first_reviewer": "A", "instructions": "verify claim"}'
            )

    def test_rejects_non_boolean_consensus(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            parse_coordinator_decision('{"consensus": "yes"}')


if __name__ == "__main__":
    unittest.main()