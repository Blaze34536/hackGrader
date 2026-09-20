import unittest

from .imports import Model
from .tools.stats import benchmark_stats, benchmark_timing


class TestScoring(unittest.TestCase):
    @benchmark_stats
    @benchmark_timing
    def test_querying(self) -> None:
        # repo: str = "https://github.com/levz0r/claude-code-statusline" # this repo is vibe coded
        repo: str = (
            "https://github.com/rexim/aoc-2023"  # this repo has no ai involvement
        )
        question: str = f"look at this repository and give it a percent numeric score based on how good the code is: {repo}"

        grader: Model = Model()
        answer: dict = grader.query(question)

        print(answer["content"])
        self.assertEqual(answer["status"], 200, "Query failure")


if __name__ == "__main__":
    unittest.main()
