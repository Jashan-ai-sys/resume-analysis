import re
import pdfplumber
import glob
import os
from collections import defaultdict, Counter


class EnhancedPDFResumeParser:
    known_companies_list = [
        'microsoft', 'google', 'amazon', 'apple', 'facebook', 'zs', 'tata',
        'infosys', 'wipro', 'deloitte', 'volvo', 'volvogroup', 'capegemini'
    ]
    professional_sections = ['professional experience', 'work experience', 'experience', 'employment history']
    skip_sections = ['skills', 'education', 'certifications', 'projects', 'summary', 'objective', 'achievements']

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self._extract_text(pdf_path)

    def _extract_text(self, pdf_path):
        txt = []
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text()
                if page_text:
                    txt.append(page_text)
        return "\n".join(txt)

    def _clean(self, line):
        # Normalize whitespace only
        return re.sub(r'\s+', ' ', line).strip()

    def contains_email_or_phone(self, text):
        email_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
        phone_pattern = re.compile(r'(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s-]?)(\d{3}[\s-]?\d{4})')
        return bool(email_pattern.search(text) or phone_pattern.search(text))

    def extract_skills(self):
        skills_list = [
        "python", "java", "javascript", "c++", "c", "sql", "excel", "analytics",
        "html", "css", "machine learning", "data analysis", "tableau", "power bi",
        "aws", "azure", "docker", "kubernetes", "c#", "pandas", "numpy",
        "tensorflow", "scikit-learn", "flask", "react", "git", "github"
        ]

        text_lower = self.text.lower()
        found = []
        for skill in skills_list:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found.append(skill)
        return sorted(set(found))

    def extract_projects(self):
        return []

    def extract_education(self):
        return []

    def extract_roles(self):
        roles_keywords = [
            "software engineer", "engineer", "analyst", "manager", "developer",
            "consultant", "intern", "specialist", "assistant", "lead"
        ]
        found = []
        for line in self.text.split('\n'):
            l = line.lower()
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', l) for keyword in roles_keywords):
                found.append(self._clean(line))
        return list(set(found))

    def extract_certifications(self):
        cert_keywords = ["AWS", "PMP", "GCP", "Azure", "TensorFlow", "Scrum", "Certified", "Oracle", "Microsoft"]
        found = []
        for line in self.text.split("\n"):
            for cert in cert_keywords:
                if cert.lower() in line.lower():
                    found.append(cert)
        return list(set(found))


    def looks_like_person_name(self, line):
        words = line.strip().split()
        if 1 <= len(words) <= 3 and all(w[0].isupper() and w.isalpha() for w in words):
            return True
        return False

    def is_bullet_line(self, line):
        bullet_regex = re.compile(r'^\s*[\u2022\u2023\u25E6\-\*\•]\s+.*', re.UNICODE)
        return bool(bullet_regex.match(line.strip()))

    def clean_line(self, line):
        line = re.sub(r'\([^)]*\)', '', line)  # Remove parentheses content
        line = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}(?:-\w+)?\b', '', line, flags=re.I)
        return line.strip(' ,.-')

    def find_known_company_in_line(self, line):
        for comp in self.known_companies_list:
            if comp.lower() in line.lower():
                return comp.title()
        return None

    def parse_company_from_sections(self, lines, skip_sections=[]):
        company_candidates = []

        for line in lines:
            ln = line.lower()
            if any(skip_section in ln for skip_section in skip_sections):
                break
            if self.is_bullet_line(line) or self.looks_like_person_name(line):
                continue

            cleaned = self.clean_line(line)
            company = self.find_known_company_in_line(cleaned)
            if company:
                return company

            if len(cleaned) < 60 and cleaned.istitle():
                company_candidates.append(cleaned)

        if company_candidates:
            return company_candidates[0]
        return None

    def extract_company(self):
        lines = [line.strip() for line in self.text.split('\n') if line.strip()]

        # Find professional experience section
        exp_start_idx = next(
            (i for i, line in enumerate(lines)
             if any(sec in line.lower() for sec in self.professional_sections)),
            None
        )

        if exp_start_idx is not None:
            next_section_idx = next(
                (i for i in range(exp_start_idx + 1, len(lines))
                 if any(sec in lines[i].lower() for sec in self.skip_sections)),
                len(lines)
            )
            section_lines = lines[exp_start_idx + 1: next_section_idx]
            company = self.parse_company_from_sections(section_lines, skip_sections=self.skip_sections)
            if company:
                return company

        # Fallback scan in entire resume text
        company = self.parse_company_from_sections(lines)
        return company


    def extract_job_role(self):
        lines = [line.strip() for line in self.text.split('\n') if line.strip()]
        current_indicators = ['present', 'current', 'till date', 'ongoing']
        title_keywords = [
            'software engineer', 'engineer', 'intern', 'analyst', 'manager', 'developer',
            'consultant', 'specialist', 'lead', 'assistant'
        ]
        locations = ['remote', 'on-site', 'hybrid', 'bangalore', 'delhi', 'mumbai', 'pune']

        for i, line in enumerate(lines):
            low_line = line.lower()
            if any(keyword in low_line for keyword in current_indicators):
                for idx in range(max(0, i - 3), min(len(lines), i + 4)):
                    candidate_line = lines[idx]
                    cleaned_line = candidate_line
                    for loc in locations:
                        cleaned_line = re.sub(fr'\b{loc}\b', '', cleaned_line, flags=re.I)
                    cleaned_line = re.sub(r'\([^)]*\)', '', cleaned_line)
                    cleaned_line = re.sub(r'\b(b\.?tech|m\.?tech|bachelor|master|phd|cgpa|gpa)[^,]*', '', cleaned_line, flags=re.I)
                    cleaned_line = cleaned_line.strip(",. -")

                    if any(tk in cleaned_line.lower() for tk in title_keywords):
                        return cleaned_line

                fallback_line = re.sub(r'\b(remote|on-site|hybrid)\b', '', line, flags=re.I).strip(",. -")
                return fallback_line

        roles = self.extract_roles()
        if roles:
            return roles[0]
        return None


def parse_all_resumes(resume_folder):
    results = []
    pdf_files = glob.glob(os.path.join(resume_folder, "*.pdf"))
    if not pdf_files:
        print("No PDF resumes found in the folder.")
        return results

    for pdf_path in pdf_files:
        candidate_id = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            parser = EnhancedPDFResumeParser(pdf_path)
            skills = parser.extract_skills()
            projects = parser.extract_projects()
            education = parser.extract_education()
            roles = parser.extract_roles()
            certifications = parser.extract_certifications()
            company = parser.extract_company()
            job_role = parser.extract_job_role()

            results.append({
                "candidate_id": candidate_id,
                "skills": skills,
                "projects": projects,
                "education": education,
                "roles": roles,
                "certifications": certifications,
                "company": company,
                "role": job_role
            })
            print(f"✔ Parsed {candidate_id}: Company: {company}, Role: {job_role}, Skills: {skills}")

        except Exception as e:
            print(f"❌ Failed to parse {candidate_id}: {e}")

    return results


def build_benchmark_by_company_role(profiles):
    benchmark = defaultdict(Counter)

    for profile in profiles:
        company = profile.get('company')
        role = profile.get('role')
        skills = profile.get('skills', [])

        if not company or not role:
            continue

        normalized_skills = [s.lower().strip() for s in skills if s]

        if normalized_skills:
            benchmark[(company.lower(), role.lower())].update(normalized_skills)

    return benchmark


def save_benchmark_to_json(benchmark, output_path):
    serializable = {
        f"{company}__{role}": dict(skill_counts)
        for (company, role), skill_counts in benchmark.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"✅ Benchmark saved to {output_path}")


