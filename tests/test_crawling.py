"""Live Devpost crawling tests for the configured example gallery.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import unittest

import pytest

from crawling.crawler import Crawler
from crawling.driver import Driver
from repositories.urls import is_github_url, repo_slug

from .tools.stats import benchmark_stats, benchmark_timing

# Live: reads a real Devpost gallery through Chrome.
pytestmark = pytest.mark.live

HACKATHON_URL = "https://2026-wics-hackathon.devpost.com"


class TestCrawling(unittest.TestCase):
    """Exercise Devpost gallery crawling and GitHub link selection."""
    def test_headless_user_agent_is_not_advertised(self) -> None:
        """Present a browser user agent, since Devpost bot-checks the headless marker."""
        driver: Driver = Driver()
        try:
            self.assertIsNotNone(driver.driver, "The Chrome session did not start")
            agent: str = driver.driver.execute_script("return navigator.userAgent")
        finally:
            driver.terminate()

        self.assertNotIn("HeadlessChrome", agent, agent)

    @benchmark_stats
    @benchmark_timing
    def test_crawling(self) -> None:
        """Collect projects and unique valid GitHub repository URLs from a gallery."""
        driver: Driver = Driver()
        try:
            crawl: Crawler = Crawler(HACKATHON_URL)
            urls: list = crawl.get_projects(driver)
            github: list = crawl.get_github_projects(urls, driver)
        finally:
            driver.terminate()

        self.assertTrue(len(urls) > 0, "No projects were crawled from the gallery")
        self.assertTrue(all(url.startswith("https://") for url in urls), urls[:5])

        self.assertTrue(len(github) > 0, "No GitHub repositories were found")
        self.assertTrue(all(is_github_url(url) for url in github), github[:5])
        self.assertEqual(
            len({repo_slug(url) for url in github}),
            len(github),
            "The same repository was collected twice",
        )


if __name__ == "__main__":
    unittest.main()
