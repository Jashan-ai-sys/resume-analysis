import re
from typing import List, Optional
from pdfminer.high_level import extract_text

class EnhancedPDFResumeParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.text = self._extract_text()
        self.lines = [line.strip() for line in self.text.split("\n") if line.strip()]

    def _extract_text(self) -> str:
        """Extract raw text from PDF."""
        try:
            return extract_text(self.filepath)
        except Exception as e:
            print(f"⚠️ PDF parsing failed: {e}")
            return ""

    # ---------------------------
    # 🔹 Company Extraction
    # ---------------------------
    def extract_company(self) -> Optional[str]:
        company_patterns = [
            r"(Deloitte|Infosys|TCS|Wipro|Accenture|Capgemini|IBM|Amazon|Microsoft|Cognizant)",
        ]
        for line in self.lines:
            for pattern in company_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0).title()
        return "Unknown"

    # ---------------------------
    # 🔹 Role Extraction
    # ---------------------------
    def extract_job_role(self) -> Optional[str]:
        role_patterns = [
            r"(Software Engineer|Senior Software Engineer|Data Scientist|Developer|Analyst|Consultant)"
        ]
        for line in self.lines:
            for pattern in role_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0).title()
        return None

    # ---------------------------
    # 🔹 Skill Extraction
    # ---------------------------
    def extract_skills(self) -> List[str]:
        skills = []
        skill_keywords = [
  "Python", "Java", "C++", "C#", "Go", "Rust", "Scala", "R", "Kotlin",
  "SQL", "NoSQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle",
  "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn",
  "Hadoop", "Spark", "Kafka", "Airflow",
  "Pandas", "Numpy", "Matplotlib", "Seaborn", "Tableau", "Power BI",
  "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins",
  "Git", "GitHub", "GitLab", "Bitbucket",
  "Linux", "Bash", "Shell Scripting", "Agile", "Scrum",
  "React", "Angular", "Vue", "Next.js", "Node.js", "Express.js",
  "HTML", "CSS", "JavaScript", "TypeScript", "Bootstrap", "Tailwind",
  "REST APIs", "GraphQL", "gRPC", "Microservices",
  "Cybersecurity", "Penetration Testing", "OAuth", "JWT", "SSL/TLS",
  "CI/CD", "DevOps", "MLOps",
  "System Design", "Algorithms", "Data Structures", "Object-Oriented Programming(OOPS)", "Design Patterns",
]

        for line in self.lines:
            for skill in skill_keywords:
                if re.search(rf"\b{skill}\b", line, re.IGNORECASE):
                    skills.append(skill)
        return sorted(set(skills))

    # ---------------------------
    # 🔹 Experience Extraction
    # ---------------------------
    def extract_experience(self):
        """
        Generate structured experience data from raw PDF lines.
        Returns a list of dicts: [{"Role": ..., "Company": ..., "Highlights": [...]}]
        """
        experience_list = []
        current_exp = {}
        highlights = []

        for line in self.lines:
            role = self.extract_job_role()
            company = self.extract_company()

            if role or company:
                if current_exp:  # save old job
                    current_exp["Highlights"] = highlights
                    experience_list.append(current_exp)
                    highlights = []

                current_exp = {
                    "Role": role or "Unknown Role",
                    "Company": company or "Unknown Company"
                }
            else:
                if line.strip():
                    highlights.append(line.strip())

        if current_exp:
            current_exp["Highlights"] = highlights
            experience_list.append(current_exp)

        return experience_list

    # ---------------------------
    # 🔹 Convert Experience → NER Input
    # ---------------------------
    def to_ner_ready_sentences(self) -> List[str]:
        """
        Convert each extracted experience into a single-sentence narrative
        for the NER model.
        """
        experiences = self.extract_experience()
        sentences = []

        for exp in experiences:
            role = exp.get("Role", "a professional")
            company = exp.get("Company", "an organization")
            skills = self.extract_skills()
            skills_str = ", ".join(skills) if skills else "various technologies"

            sentence = f"Worked as a {role} at {company} with expertise in {skills_str}."
            sentences.append(sentence)

        return sentences
