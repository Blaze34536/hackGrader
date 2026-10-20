"""Coordinate gallery crawling, repository grading, and score persistence.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
from typing import Any

from crawling.crawler import Crawler
from crawling.driver import Driver
from grading.reporting import Reporting
from grading.runtime import BASE_QUESTION, Model
from repositories.urls import repo_slug


def individual_query(url: str, grader: Model) -> dict[str, Any]:
    """Evaluate one repository with the shared grader instance."""
    return grader.query_with_retry(BASE_QUESTION.format(url=url))


def run_gallery(url: str) -> dict[str, dict[str, Any]]:
    """Crawl one gallery, preflight the grader, and save each completed evaluation."""
    reporter = Reporting()
    driver = Driver()
    try:
        crawler = Crawler(url)
        projects = crawler.get_projects(driver)
        print(f"Found {len(projects)} project(s) in the gallery.")
        github_projects = crawler.get_github_projects(projects, driver)
    finally:
        driver.terminate()

    if not github_projects:
        raise SystemExit(f"No GitHub repositories were found for {url}. Nothing to grade.")

    print(f"Grading {len(github_projects)} repository(ies).\n")
    grader = Model()
    readiness_error = grader.preflight_error()
    if readiness_error:
        raise SystemExit(f"[!] {readiness_error}")

    scores: dict[str, dict[str, Any]] = {}
    for project in github_projects:
        slug = repo_slug(project)
        if slug in scores:
            continue
        print(f"grading: {slug}\n")
        try:
            scores[slug] = individual_query(project, grader)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            scores[slug] = {
                "status": 500,
                "content": f"Unhandled error while grading {project}: {exc}",
            }
        with open(reporter.json_path, "w", encoding="utf-8") as file:
            json.dump(scores, file, ensure_ascii=False, indent=4)

    reporter.generate_report()
    return scores
