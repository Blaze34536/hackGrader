import unittest, sys, json
from .imports import Model, Driver, Crawler, Reporting

from .tools.stats import benchmark_stats, benchmark_timing

class TestPipeline(unittest.TestCase):
    @benchmark_stats
    def test_pipeline(self) -> None:
        url: str = "https://2026-wics-hackathon.devpost.com"
        driver: Driver = Driver()
        crawler: Crawler = Crawler(url)
        reporter: Reporting = Reporting()
        projects: list = crawler.get_projects(driver)
        github_projects: list = crawler.get_github_projects(projects, driver)

        all_scores = {}
        grader = Model()

        for project in github_projects:
            if not project: 
                continue

            with self.subTest(project_url=project):
                result = self.individual_query(project, grader)

                parts = project.rstrip('/').split('/')
                if len(parts) >= 2:
                    owner = parts[-2]
                    print(f"grading: {owner}")
                    all_scores[owner] = result

        with open("scores.json", 'w', encoding="utf-8") as file:
            json.dump(all_scores, file, ensure_ascii=False, indent=4)

        reporter.generate_report()

    @benchmark_timing
    def individual_query(self, url: str, model: Model) -> None:
        question: str = f"Please provide a concise code review for the following url that includes two percentage scores: The estimated percentage of AI involvement in generating the code. A code goodness score based on the following criteria: cleanliness, readability, functionality, maintainability, scalability, and documentation.Summarize your evaluation clearly and succinctly, highlighting the strengths and areas for improvement in the code. {url}"
        answer: dict = model.query(question)

        print(answer['content'])
        self.assertEqual(answer["status"], 200, "Query failure")
        return answer["content"]

if __name__ == "__main__":
    unittest.main()
