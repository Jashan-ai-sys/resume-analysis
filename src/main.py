import torch
import json
import re
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import T5ForConditionalGeneration, T5TokenizerFast, pipeline
from enhanced_parser import EnhancedPDFResumeParser
from difflib import get_close_matches
from fastapi.middleware.cors import CORSMiddleware

# ===============================
# 1️⃣ Load Models & Skill JSON
# ===============================
MODEL_DIR = r"C:/Users/WIN11/resume-analysis/model/t5_skill_feedback"
SKILL_DICT_FILE = r"C:/Users/WIN11/resume-analysis/data/skill_requirement_dataset.json"
NER_MODEL_PATH = r"C:/Users/WIN11/resume-analysis/model/fourth_phase_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load T5
tokenizer = T5TokenizerFast.from_pretrained(MODEL_DIR)
t5_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

# Load NER
ner_model = pipeline("ner", model=NER_MODEL_PATH, tokenizer=NER_MODEL_PATH)

# Load skill dictionary
with open(SKILL_DICT_FILE, "r", encoding="utf-8") as f:
    skill_data = json.load(f)

skill_dict = {}
for company, roles in skill_data.items():
    for role, skills in roles.items():
        skill_dict[f"{company}|{role}"] = skills

# ===============================
# 2️⃣ Helper Functions
# ===============================
def normalize_string(s: str):
    return s.strip().title() if s else ""
# 1. Convert parsed data into narrative


# 2. Run NER on clean narrative

def extract_skills(text: str):
    """Run NER on resume text to extract skills"""
    raw_entities = ner_model(text)
    skills = []
    for ent in raw_entities:
        if ent["entity"].upper().startswith("SKILL"):
            word = ent["word"].strip()
            if word.startswith("##"):
                word = word.replace("##", "")
            skills.append(word)
    return list({s.capitalize() for s in skills if s})

def generate_feedback(company, role, candidate_skills):
    input_text = f"Company: {company} | Role: {role} | Candidate Skills: {', '.join(candidate_skills)}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = t5_model.generate(
        inputs,
        max_length=128,
        num_beams=4,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_required_skills(company: str, role: str):
    """Lookup JSON skills with normalization & fuzzy matching"""
    company_norm = normalize_string(company)
    role_norm = normalize_string(role)
    lookup_key = f"{company_norm}|{role_norm}"

    if lookup_key in skill_dict:
        return skill_dict[lookup_key]

    matches = get_close_matches(lookup_key, skill_dict.keys(), n=1, cutoff=0.7)
    if matches:
        return skill_dict[matches[0]]

    return []

def extract_t5_suggestions(raw_output: str):
    """Extract T5-suggested skills from its output"""
    suggestions = []
    patterns = [
        r'add ([^.]*)\.',                 
        r'consider strengthening ([^.]*)\.',  
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw_output, flags=re.IGNORECASE)
        for match in matches:
            skills = [s.strip() for s in match.split(',')]
            suggestions.extend(skills)
    return list({s.capitalize() for s in suggestions if s})
def clean_t5_suggestions(t5_skills):
    replacements = {
        "Cloud deployment for better fit": "Cloud Deployment",
        "Data structures": "Data Structures",
        "Debugging for better fit": "Debugging",
        "System design": "System Design",
        "APIs": "APIs"
    }
    cleaned = [replacements.get(s.strip(), s.strip().title()) for s in t5_skills]
    return set(cleaned)

RESOURCES_FILE = r"C:/Users/WIN11/resume-analysis/data/resource_list.json"

with open(RESOURCES_FILE, "r", encoding="utf-8") as f:
    learning_resources = json.load(f)
# ===============================
# Skill Normalization Map
# ===============================
skill_normalization = {
    # Programming Languages
    "c++": "c++",
    "c#": "csharp",
    "c": "c",
    "js": "javascript",
    "javascript": "javascript",
    "typescript": "typescript",
    "python": "python",
    "java": "java",
    "sql": "sql",
    "sql server": "databases",
    "nosql": "databases",
    "mongodb": "databases",
    "kql": "kql",

    # Web Frameworks & Libraries
    "react.js": "react",
    "reactjs": "react",
    "react": "react",
    "angular": "angular",
    "next.js": "nextjs",
    "node.js": "node",
    "node": "node",
    "express.js": "express",
    "express": "express",
    "redux": "redux",
    "ngrx": "ngrx",
    "rxjs": "rxjs",

    # Tools
    "git": "git",
    "github": "git",
    "vs code": "visual studio code",
    "visual studio": "visual studio",
    "eclipse": "eclipse",
    "postman": "postman",

    # Web Technologies
    "html": "html",
    "html5": "html",
    "css": "css",
    "css3": "css",
    "bootstrap": "bootstrap",
    "webpack": "webpack",
    "babel": "babel",

    # Cloud & DevOps
    "azure": "azure",
    "aws": "aws",
    "docker": "docker",
    "kubernetes": "kubernetes",

    # Security & APIs
    "restful apis": "apis",
    "apis": "apis",
    "oauth": "oauth 2.0",
    "jwt": "jwt",
    "web sockets": "web sockets",

    # Concepts
    "oop": "oop",
    "object oriented programming": "oop",
    "system design": "system design",
    "algorithms": "algorithms",
    "data structures": "data structures",
    "databases": "databases",
    "debugging": "debugging",
    "testing": "testing",
    "agile": "agile",
    "cloud deployment": "cloud deployment"
}

def normalize_skill(skill: str) -> str:
    """Map skill to a canonical form using skill_normalization dict."""
    if not skill:
        return ""
    skill = skill.strip().lower()
    return skill_normalization.get(skill, skill)

def post_process_feedback(company, role, candidate_skills, raw_output):
    required_skills = get_required_skills(company, role)
    print(f"\n🔹 JSON required skills for {company}|{role}: {required_skills}")

    # T5 suggested skills
    t5_skills = clean_t5_suggestions(extract_t5_suggestions(raw_output))
    print(f"🔹 T5 suggested skills: {t5_skills}")

    # Merge required + T5
    all_required_skills = list(set(required_skills).union(t5_skills))

    # Normalize both sides
    cand_set = {normalize_skill(s) for s in candidate_skills}
    req_set = {normalize_skill(s) for s in all_required_skills}

    # Compute missing
    missing = [s for s in all_required_skills if normalize_skill(s) not in cand_set]

    # Match %
    match_percent = round(len(req_set & cand_set) / len(req_set) * 100, 2) if req_set else 0

    # Collect resources
    resources_for_missing = {
        skill: learning_resources.get(skill, {"description": "No resources available", "resources": []})
        for skill in missing
    }

    feedback = (
        f"✅ You already have: {', '.join(sorted(candidate_skills))}.\n"
        f"📌 To improve your profile for {role}, focus on: {', '.join(missing) if missing else 'No extra skills needed!'}.\n"
        f"📊 Profile Match: {match_percent}%"
    )

    return {
        "candidate_skills": sorted(candidate_skills),
        "required_skills": sorted(all_required_skills),
        "missing_skills": sorted(missing),
        "match_percent": match_percent,
        "raw_output": raw_output,
        "final_feedback": feedback,
        "learning_resources": resources_for_missing
    }

def build_narrative(name: str, role: str, company: str, skills: list, text: str) -> str:
        """
        Convert structured resume fields into a natural language narrative.
        Falls back to raw resume text if key info is missing.
        """
        # Clean values
        name = name or "The candidate"
        role = role or "a professional"
        company = company or "the company"
        skills = skills or []

        skills_str = ", ".join(skills) if skills else "various technical skills"

        narrative = (
            f"{name} is employed as {role} at {company}. "
            f"{name.split()[0]} has strong skills in {skills_str}. "
        )

        # Add fallback raw text if parser missed details
        if text and len(text) > 50:
            narrative += f"\nAdditional details from the resume: {text[:1000]}"

        return narrative



# ===============================
# 3️⃣ FastAPI App
# ===============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all during dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_resume/")
async def analyze_resume(file: UploadFile, company: str = Form(...), role: str = Form(...)):
    try:
        # Always initialize defaults
        parser_skills, ner_skills, candidate_skills = [], [], []

        # Save file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        parser = EnhancedPDFResumeParser(temp_file)
        resume_text = parser.text
        parser_skills = parser.extract_skills()
        parser_company = parser.extract_company() or company
        parser_role = parser.extract_job_role() or role

        norm_company = normalize_string(parser_company)
        norm_role = normalize_string(parser_role)

        narrative_text = build_narrative(
            name=None,
            role=parser_role,
            company=parser_company,
            skills=parser_skills,
            text=resume_text
        )

        # Run NER
        ner_skills = extract_skills(narrative_text)

        # 🔹 Merge both parser + NER skills
        candidate_skills = list(
            {normalize_skill(s) for s in (parser_skills + ner_skills) if s}
        )

        print("🔹 Parser skills:", parser_skills)
        print("🔹 NER skills:", ner_skills)
        print("🔹 Final candidate_skills:", candidate_skills)

        # T5 feedback
        raw = generate_feedback(norm_company, norm_role, candidate_skills)

        # Post-process
        result = post_process_feedback(norm_company, norm_role, candidate_skills, raw)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ===============================
# 4️⃣ Run server
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
