import sys, json
from crawler import Crawler
from driver import Driver
from model import Model

def main() -> None:
    url_args = sys.argv[1:]
    if not url_args:
        raise Exception("Please pass in a URL as an argument")

    url: str = url_args[0]

    driver: Driver = Driver()
    crawler: Crawler = Crawler(url)
    projects: list = crawler.get_projects(driver)
    github_projects: list = crawler.get_github_projects(projects, driver)

    all_scores = {}
    grader = Model()

    for project in github_projects:
        if not project:
            continue
        
        parts = project.rstrip('/').split('/')
        if len(parts) >= 2:
            owner = parts[-2]
            result = individual_query(project, grader)
            all_scores[owner] = result

    with open("scores.json", 'w', encoding="utf-8") as file:
        json.dump(all_scores, file, ensure_ascii=False, indent=4)

def individual_query(url: str, grader: Model) -> dict:
    question: str = (
        f"Please review the hackathon repository at the following URL: {url}. "
        "Provide your evaluation including the two percentage scores and the concise summary "
        "of strengths and weaknesses as instructed."
    )
    answer = grader.query(question)
    return answer

if __name__ == "__main__":
    main()