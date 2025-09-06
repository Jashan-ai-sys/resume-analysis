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
        """
        Extract company name by checking lines near job roles.
        Defaults to 'Unknown' if not found.
        """
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
        """
        Extract job role from lines (common role keywords).
        """
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
        """
        Simple regex + keyword-based skill extraction.
        """
        skills = []
        skill_keywords = [
            "Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning", "TensorFlow",
            "PyTorch", "Data Visualization", "Pandas", "Numpy", "Spark", "Cloud", "AWS",
            "Azure", "Docker", "Kubernetes", "Git", "Tableau", "Statistics"
        ]
        for line in self.lines:
            for skill in skill_keywords:
                if re.search(rf"\b{skill}\b", line, re.IGNORECASE):
                    skills.append(skill)
        return sorted(set(skills))

    # ---------------------------
    # 🔹 Raw Fallback
    # ---------------------------
    def get_text(self) -> str:
        return self.text
