import sys

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.webdriver import WebDriver
from webdriver_manager.chrome import ChromeDriverManager


class Driver:
    def __init__(self) -> None:
        chromeOptions = Options()
        chromeOptions.add_argument("--headless=new")
        chromeOptions.add_argument("--disable-gpu")
        chromeOptions.add_argument("--disable-renderer-backgrounding")
        chromeOptions.add_argument("--disable-background-timer-throttling")
        chromeOptions.add_argument("--disable-backgrounding-occluded-windows")
        chromeOptions.add_argument("--disable-dev-shm-usage")
        chromeOptions.add_argument("--no-sandbox")

        if sys.platform.startswith("linux"):
            self.driver: WebDriver | None = self.linuxWebdriver(chromeOptions)
        self.driver: WebDriver | None = webdriver.Chrome(options=chromeOptions)

    def linuxWebdriver(self, chromeOptions) -> None:
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=chromeOptions
        )

    def connectUrl(self, url: str) -> bool:
        try:
            self.driver.get(url)  # type: ignore
            return True
        except:
            return False

    def terminate(self) -> None:
        self.driver.quit()  # type: ignore
