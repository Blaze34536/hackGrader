from selenium.webdriver.common.by import By

import driver


class Crawler:
    def __init__(self, link: str) -> None:
        self.base_url: str = link
        self.gallery_url: str = self.base_url + "/project-gallery"
        self.queue: list = list()

    def get_projects(self, webdriver: driver.Driver) -> list:
        projects: list = list()
        index: int = 1
        next_page: bool = True
        primary_driver = webdriver.driver

        while next_page:
            webdriver.connectUrl(self.gallery_url + f"?page={index}")
            assert primary_driver is not None
            elements = primary_driver.find_elements(By.CLASS_NAME, "link-to-software")
            page_hrefs = [elem.get_attribute("href") for elem in elements]

            if not page_hrefs or all(href in projects for href in page_hrefs):
                next_page = False
            else:
                for href in page_hrefs:
                    if href not in projects:
                        projects.append(href)
            index += 1

        return projects

    def crawl_project(self, project_url: str, webdriver: driver.Driver) -> str:
        webdriver.connectUrl(project_url)
        primary_driver = webdriver.driver

        try:
            assert primary_driver is not None
            element = primary_driver.find_element(
                By.CSS_SELECTOR, "ul[data-role='software-urls'] a"
            )
            url = element.get_attribute("href")
            return str(url)
        except:
            return ""

    def get_github_projects(self, project_urls: list, webdriver: driver.Driver) -> list:
        github_urls: list = list()
        for project in project_urls:
            if project == "":
                continue
            github_urls.append(self.crawl_project(project, webdriver))

        return github_urls
