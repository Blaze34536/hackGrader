import unittest

from .tools.stats import benchmark_timing, benchmark_stats
from .imports import Driver, Crawler

class TestCrawling(unittest.TestCase):
    @benchmark_stats
    @benchmark_timing
    def test_crawling(self) -> None:
        url: str = "https://2026-wics-hackathon.devpost.com"
        driver: Driver = Driver()
        crawl: Crawler = Crawler(url)

        urls: list = crawl.get_projects(driver)
        github: list = crawl.get_github_projects(urls, driver)

        self.assertEqual(len(github) > 0, True, "Query failure")

if __name__ == "__main__":
    unittest.main()
