"""Discover Devpost projects and their GitHub repository links.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from urllib.parse import urljoin

from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver as SeleniumWebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from repositories.urls import is_github_url, repo_slug

from .driver import Driver

_BOT_CHECK_MARKERS = ("just a moment", "performing security verification")


class Crawler:
    """Walk a Devpost project gallery and collect unique GitHub repositories."""

    def __init__(self, link: str, page_timeout: float = 20.0, max_pages: int = 60) -> None:
        """Set a gallery URL and bounded browser wait/page limits."""
        self.base_url = link.rstrip("/")
        self.gallery_url = f"{self.base_url}/project-gallery"
        self.page_timeout = page_timeout
        self.max_pages = max_pages

    def get_projects(self, webdriver: Driver) -> list[str]:
        """Read project links from each gallery page until the listing ends."""
        primary_driver = webdriver.driver
        assert primary_driver is not None, "WebDriver is not initialised"
        wait = WebDriverWait(primary_driver, self.page_timeout)
        projects: list[str] = []
        seen: set[str] = set()

        for index in range(1, self.max_pages + 1):
            if not webdriver.connectUrl(f"{self.gallery_url}?page={index}"):
                break
            try:
                wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "link-to-software"))
                )
            except TimeoutException:
                if index == 1:
                    if self._awaiting_bot_check(primary_driver):
                        raise RuntimeError(
                            f"Devpost answered {self.gallery_url}?page=1 with a bot "
                            "verification page instead of the gallery, so no projects "
                            "could be read. Retry, or raise page_timeout if the check "
                            "is slow to clear."
                        ) from None
                    raise RuntimeError(
                        f"No project links found at {self.gallery_url}?page=1. "
                        "Check the Devpost URL, or the page markup may have changed."
                    ) from None
                break

            hrefs = [
                element.get_attribute("href")
                for element in primary_driver.find_elements(By.CLASS_NAME, "link-to-software")
            ]
            fresh = [urljoin(self.base_url, href) for href in hrefs if href]
            fresh = [href for href in fresh if href not in seen]
            if not fresh:
                break
            projects.extend(fresh)
            seen.update(fresh)

        if not projects:
            raise RuntimeError(f"No projects were found in the gallery at {self.gallery_url}.")
        return projects

    @staticmethod
    def _awaiting_bot_check(browser: SeleniumWebDriver) -> bool:
        """Report whether the browser is sitting on a bot-verification interstitial."""
        try:
            page = browser.page_source.lower()
        except WebDriverException:
            return False
        return any(marker in page for marker in _BOT_CHECK_MARKERS)

    def crawl_project(self, project_url: str, webdriver: Driver) -> str:
        """Open one Devpost project and return its preferred GitHub link."""
        primary_driver = webdriver.driver
        if primary_driver is None or not project_url or not webdriver.connectUrl(project_url):
            return ""

        wait = WebDriverWait(primary_driver, self.page_timeout)
        try:
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "ul[data-role='software-urls'] a")
                )
            )
        except TimeoutException:
            return ""

        links = [
            href
            for href in (
                element.get_attribute("href")
                for element in primary_driver.find_elements(
                    By.CSS_SELECTOR, "ul[data-role='software-urls'] a"
                )
            )
            if href
        ]
        return next((link for link in links if is_github_url(link)), links[0] if links else "")

    def get_github_projects(self, project_urls: list[str], webdriver: Driver) -> list[str]:
        """Collect unique GitHub URLs while reporting Devpost entries without one."""
        github_urls: list[str] = []
        seen: set[str] = set()
        skipped: list[str] = []
        for project in project_urls:
            if not project:
                continue
            url = self.crawl_project(project, webdriver)
            if not is_github_url(url):
                skipped.append(project)
                continue
            slug = repo_slug(url)
            if slug not in seen:
                seen.add(slug)
                github_urls.append(url)

        if skipped:
            print(f"[i] Skipped {len(skipped)} project(s) without a GitHub link:")
            for project in skipped[:10]:
                print(f"    {project}")
            if len(skipped) > 10:
                print(f"    ... and {len(skipped) - 10} more")
        return github_urls
