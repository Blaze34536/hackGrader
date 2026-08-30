import sys
import json
from urllib.parse import urlparse  

from crawler import Crawler
from driver import Driver
from model import Model
from reporting import Reporting

def main() -> None:
    reporter: Reporting = Reporting()
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
        
        print(f"project: {project}\n")
        
        parsed_url = urlparse(project)
        path_parts = [p for p in parsed_url.path.split('/') if p]
        
        if len(path_parts) >= 1:
            owner = path_parts[0]  
            print(f"grading: {owner}\n")
            
            result = individual_query(project, grader)
            
            all_scores[owner] = result
            
            with open("scores.json", 'w', encoding="utf-8") as file:
                json.dump(all_scores, file, ensure_ascii=False, indent=4)

    reporter.generate_report()
    
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