import json
import textwrap

from jinja2 import Template
from xhtml2pdf import pisa

with open("scores.json", "r", encoding="utf-8") as file:
    data = json.load(file)

clean_data: dict = dict()
for project, details in data.items():
    if isinstance(details, dict):
        text = str(details.get("content", details))
    else:
        text = str(details)

    wrapped_lines = []
    for line in text.split("\n"):
        wrapped_line = textwrap.fill(line, width=75, break_long_words=True)
        wrapped_lines.append(wrapped_line)

    clean_data[project] = "\n".join(wrapped_lines)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        @page {
          size: letter portrait;
          margin: 1.2in 1.2in 1.2in 1.2in;
        }

        body { 
          font-family: Helvetica, sans-serif; 
          font-size: 16px;
          color: #333; 
        }

        h1 { 
          text-align: center; 
          font-size: 26px;
          color: #2c3e50; 
          margin-bottom: 25px; 
        }

        .project-title { 
          text-align: center;
          font-size: 22px;   
          font-weight: bold; 
          color: #2980b9; 
          margin-bottom: 20px; 
        }
        
        .content { 
          font-size: 16px;   
          white-space: pre-wrap; 
          line-height: 1.6;  
          width: 100%; 
        }
        .page-break { page-break-before: always; }
    </style>
</head>
<body>
    <h1>Hackathon Code Review Report</h1>
    {% for project, details in data.items() %}
    <div class="{% if not loop.first %}page-break{% endif %}">
        <div class="project-title">Project: {{ project }}</div>
        <div class="content">{{ details }}</div>
    </div>
    {% endfor %}
</body>
</html>
"""

template = Template(html_template)
rendered_html = template.render(data=clean_data)

with open("report.pdf", "wb") as output:
    pisa.CreatePDF(rendered_html, dest=output)
