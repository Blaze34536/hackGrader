"""Live end-to-end checks for gallery scoring and report persistence.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import json
import os
import tempfile
import unittest

import pytest

from contracts import is_valid_report
from crawling.crawler import Crawler
from crawling.driver import Driver
from grading.reporting import Reporting
from grading.runtime import Model
from repositories.urls import repo_slug

from .tools.stats import benchmark_stats, benchmark_timing

# Live: crawls the gallery and grades through the served model.
pytestmark = pytest.mark.live

HACKATHON_URL = "https://2026-wics-hackathon.devpost.com"
SAMPLE_SIZE = 2


class TestPipeline(unittest.TestCase):
    """Exercise gallery crawling, repository grading, and persisted result reports."""
    @benchmark_stats
    @benchmark_timing
    def test_pipeline(self) -> None:
        """Crawl a gallery, grade samples, and verify the saved report annotations."""
        driver: Driver = Driver()
        try:
            crawler: Crawler = Crawler(HACKATHON_URL)
            projects: list = crawler.get_projects(driver)
            github_projects: list = crawler.get_github_projects(projects, driver)
        finally:
            driver.terminate()

        self.assertGreater(len(projects), 0, "No Devpost projects were crawled")
        self.assertGreater(len(github_projects), 0, "No GitHub repositories were found")

        sample = github_projects[:SAMPLE_SIZE]
        grader: Model = Model()
        all_scores: dict = {}

        for project in sample:
            with self.subTest(project_url=project):
                result: dict = self.individual_query(project, grader)

                self.assertEqual(result["status"], 200, f"Query failure for {project}")
                self.assertTrue(
                    is_valid_report(result["content"]),
                    f"Unparseable report for {project}",
                )
                all_scores[repo_slug(project)] = result

        self.assertEqual(
            len({entry["content"] for entry in all_scores.values()}),
            len(all_scores),
            "Two different repositories received identical reports",
        )

        with tempfile.TemporaryDirectory() as tmp:
            reporter = Reporting(json_path=os.path.join(tmp, "scores.json"))
            with open(reporter.json_path, "w", encoding="utf-8") as file:
                json.dump(all_scores, file, ensure_ascii=False, indent=4)

            reporter.generate_report()

            with open(reporter.json_path, encoding="utf-8") as file:
                written = json.load(file)

        for name, entry in written.items():
            self.assertEqual(entry["status"], 200, f"{name} failed grading")
            self.assertIsInstance(entry["ai_score"], int, f"{name} has no judge AI score")
            self.assertIsInstance(entry["code_score"], int, f"{name} has no judge quality score")
            self.assertNotIn("flags", entry)

    @benchmark_timing
    def individual_query(self, url: str, model: Model) -> dict:
        """Request and print one concise repository review from the shared grader."""
        question: str = (
            "Please provide a concise code review for the following url that includes two "
            "percentage scores: The estimated percentage of AI involvement in generating the "
            "code. A code goodness score based on the following criteria: cleanliness, "
            "readability, functionality, maintainability, scalability, and documentation."
            f"Summarize your evaluation clearly and succinctly. {url}"
        )
        answer: dict = model.query_with_retry(question)

        print(answer["content"])
        return answer


if __name__ == "__main__":
    unittest.main()
