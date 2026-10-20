"""Manage the Selenium Chrome session used to navigate Devpost pages.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import sys

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.webdriver import WebDriver as SeleniumWebDriver
from webdriver_manager.chrome import ChromeDriverManager

_HEADLESS_MARKER = "HeadlessChrome"


class Driver:
    """Create, navigate, and close the Chrome driver for one crawl."""

    def __init__(self, headless: bool = True) -> None:
        """Start a Chrome session, optionally without a visible browser window."""
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-renderer-backgrounding")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1440,1000")
        self.driver: SeleniumWebDriver | None = self._build(options)
        self._present_browser_user_agent()

    @staticmethod
    def _build(options: Options) -> SeleniumWebDriver:
        """Start Chrome with a platform-appropriate driver service."""
        if sys.platform.startswith("linux"):
            service = Service(ChromeDriverManager().install())
            return webdriver.Chrome(service=service, options=options)
        return webdriver.Chrome(options=options)

    def _present_browser_user_agent(self) -> None:
        """Replace the headless marker in the user agent before the first navigation.

        Chrome advertises ``HeadlessChrome`` in both the request header and
        ``navigator.userAgent``, and Devpost's managed bot check answers that token with
        a verification interstitial instead of the gallery, so the crawler never sees
        the markup it waits for. Presenting the browser's own user agent clears the
        check; hiding ``navigator.webdriver`` alone does not.
        """
        if self.driver is None:
            return
        try:
            advertised = self.driver.execute_script("return navigator.userAgent")
            if not isinstance(advertised, str) or _HEADLESS_MARKER not in advertised:
                return
            self.driver.execute_cdp_cmd(
                "Network.setUserAgentOverride",
                {"userAgent": advertised.replace(_HEADLESS_MARKER, "Chrome")},
            )
        except WebDriverException as exc:
            print(
                "[!] Could not present the browser's own user agent; Devpost may serve "
                f"a bot check instead of the gallery: {str(exc).splitlines()[0]}"
            )

    def connectUrl(self, url: str) -> bool:
        """Navigate to a URL and return false when the browser rejects the request."""
        if self.driver is None:
            return False
        try:
            self._discard_bot_protection_cookies()
            self.driver.get(url)
            return True
        except WebDriverException as exc:
            print(f"[!] Could not load {url}: {str(exc).splitlines()[0]}")
            return False

    def _discard_bot_protection_cookies(self) -> None:
        """Drop the previous page's bot-protection cookies before the next navigation.

        Devpost answers every page load after the first one in a session with a
        Cloudflare verification page that never clears, because the ``cf_clearance`` and
        ``aws-waf-token`` cookies earned by the first page are replayed for pages that
        never earned them. Clearing cookies here gets each navigation through in one to
        two seconds; the WebDriver ``delete_all_cookies`` command does not, since it
        leaves the network-layer cookies behind.
        """
        if self.driver is None:
            return
        try:
            self.driver.execute_cdp_cmd("Network.clearBrowserCookies", {})
        except WebDriverException as exc:
            print(
                "[!] Could not clear bot-protection cookies before navigating: "
                f"{str(exc).splitlines()[0]}"
            )

    def terminate(self) -> None:
        """Close the browser once and clear the reference even if shutdown fails."""
        if self.driver is not None:
            try:
                self.driver.quit()
            finally:
                self.driver = None
