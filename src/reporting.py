import json, os, re

class Reporting:
    def __init__(self):
        self.json_path = "scores.json"
        self.ai_threshold = 20
        self.exceptional_threshold = 90
        self.broken_threshold = 40

    def generate_report(self) -> None:
        if not os.path.exists(self.json_path):
            raise Exception(f"Error: The file {self.json_path} was not found.")

        with open(self.json_path, 'r', encoding="utf-8") as file:
            try:
                projects = json.load(file)
            except json.JSONDecodeError:
                raise Exception("Error: Invalid JSON format.")

        for _, project in projects.items():
            content = project.get("content", "")
            
            ai_match = re.search(r"AI Involvement Score:\s*(\d+)%", content, re.IGNORECASE)
            ai_score = int(ai_match.group(1)) if ai_match else 0
            
            code_match = re.search(r"Code Quality Score:\s*(\d+)%", content, re.IGNORECASE)
            code_score = int(code_match.group(1)) if code_match else 0
            
            summary_match = re.search(r"3\.\s*Summary & Notable Findings:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
            model_summary = summary_match.group(1).strip() if summary_match else ""
            
            flags = []
            notable_findings = []

            if ai_score > self.ai_threshold:
                flags.append("HIGH_AI_USAGE")
                notable_findings.append(f"Flagged for AI usage: Score of {ai_score}% exceeds the conservative baseline of {self.ai_threshold}%.")
            
            if code_score >= self.exceptional_threshold:
                flags.append("EXCEPTIONAL_PROJECT")
                notable_findings.append(f"Flagged for Quality: Score of {code_score}% indicates a highly polished and exceptional project.")
            elif code_score <= self.broken_threshold:
                flags.append("NEEDS_REVIEW_POOR_QUALITY")
                notable_findings.append(f"Flagged for Quality: Score of {code_score}% suggests an empty shell or severe syntax errors.")

            if model_summary:
                notable_findings.append(f"Model Summary: {model_summary}")

            project["flags"] = flags
            project["report_findings"] = notable_findings
            
            # temps for sorting
            project["_ai_score"] = ai_score
            project["_code_score"] = code_score

        sorted_items = sorted(
            projects.items(),
            key=lambda item: (
                len(item[1].get("flags", [])) == 0, 
                min(item[1].get("_ai_score", 0), item[1].get("_code_score", 0))
            )
        )

        final_projects = {}
        for name, project in sorted_items:
            project.pop("_ai_score", None)
            project.pop("_code_score", None)
            final_projects[name] = project

        with open(self.json_path, 'w', encoding="utf-8") as file:
            json.dump(final_projects, file, ensure_ascii=False, indent=4)
            
        print(f"Report successfully generated and appended to {self.json_path}.")