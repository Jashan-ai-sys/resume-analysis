import torch
import json
import re
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import T5ForConditionalGeneration, T5TokenizerFast, pipeline
from enhanced_parser import EnhancedPDFResumeParser
from difflib import get_close_matches

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

def post_process_feedback(company, role, candidate_skills, raw_output):
    # JSON required skills
    required_skills = get_required_skills(company, role)
    print(f"\n🔹 JSON required skills for {company}|{role}: {required_skills}")

    # T5 suggested skills
    t5_skills = clean_t5_suggestions(extract_t5_suggestions(raw_output))
    print(f"🔹 T5 suggested skills: {t5_skills}")

    # Merge both
    all_required_skills = list(set(required_skills).union(t5_skills))
    print(f"🔹 All required skills (merged JSON + T5): {all_required_skills}")

    # Compute missing skills
    normalize = lambda s: s.strip().lower()
    cand_set = {normalize(s) for s in candidate_skills}
    req_set = {normalize(s) for s in all_required_skills}
    missing = [s for s in all_required_skills if normalize(s) not in cand_set]
    print(f"🔹 Missing skills: {missing}")

    # Match percentage
    match_percent = round(len(req_set & cand_set) / len(req_set) * 100, 2) if req_set else 0
    print(f"🔹 Match percent: {match_percent}%")
    resources_for_missing = {}
    for skill in missing:
        resources_for_missing[skill] = learning_resources.get(skill, {"description": "No resources available", "resources": []})

    # Feedback text
    feedback = (
        f"✅ You already have: {', '.join(candidate_skills)}.\n"
        f"📌 To improve your profile for {role}, focus on: {', '.join(missing) if missing else 'No extra skills needed!'}.\n"
        f"📊 Profile Match: {match_percent}%"
    )

    return {
        "candidate_skills": candidate_skills,
        "required_skills": all_required_skills,
        "missing_skills": missing,
        "match_percent": match_percent,
        "raw_output": raw_output,
        "final_feedback": feedback,
        "learning_resources": resources_for_missing
    }


# ===============================
# 3️⃣ FastAPI App
# ===============================
app = FastAPI()

@app.post("/analyze_resume/")
async def analyze_resume(file: UploadFile, company: str = Form(...), role: str = Form(...)):
    try:
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

        ner_skills = extract_skills(resume_text)
        candidate_skills = list({s.capitalize() for s in parser_skills + ner_skills})

        # T5 feedback
        raw = generate_feedback(norm_company, norm_role, candidate_skills)

        # Post-process (merge JSON + T5)
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
