import torch
import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import T5ForConditionalGeneration, T5TokenizerFast, pipeline
from enhanced_parser import EnhancedPDFResumeParser
from difflib import get_close_matches

# ===============================
# 🔹 1. Load Models
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

# ===============================
# 🔹 2. Load Skill Dictionary
# ===============================
with open(SKILL_DICT_FILE, "r", encoding="utf-8") as f:
    skill_data = json.load(f)

from difflib import get_close_matches

def debug_required_skills_lookup(company, role, skill_dict):
    company_norm = company.strip().title()
    role_norm = role.strip().title()
    lookup_key = f"{company_norm}|{role_norm}"

    print("\n🔹 Debugging required skills lookup")
    print("Lookup key:", repr(lookup_key))
    print("Available keys in skill_dict:", list(skill_dict.keys())[:10], "...")  # first 10 keys

    # Exact match
    if lookup_key in skill_dict:
        print("✅ Exact match found!")
        return skill_dict[lookup_key]

    # Fuzzy match
    matches = get_close_matches(lookup_key, skill_dict.keys(), n=3, cutoff=0.6)
    if matches:
        print("⚠️ Fuzzy matches found:", matches)
        print(f"Using closest match: {matches[0]}")
        return skill_dict[matches[0]]

    print("❌ No match found for this company|role.")
    return []

skill_dict = {}
for company, roles in skill_data.items():
    for role, skills in roles.items():
        skill_dict[f"{company}|{role}"] = skills

# ===============================
# 🔹 3. NER Extraction + Filtering
# ===============================
def extract_skills(text: str):
    """Run NER on text and return a cleaned skill list"""
    raw_entities = ner_model(text)
    skills = []
    for ent in raw_entities:
        if ent["entity"].upper().startswith("SKILL"):
            word = ent["word"].strip()
            if word.startswith("##"):
                word = word.replace("##", "")
            skills.append(word)
    return list({s.capitalize() for s in skills if s})

# ===============================
# 🔹 4. T5 Feedback Generator
# ===============================
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

# ===============================
# 🔹 5. Robust Skill Lookup with Fuzzy Matching
# ===============================
def get_required_skills(company: str, role: str):
    """
    Find required skills for a company|role key.
    Uses fuzzy matching if exact key not found.
    """
    company_norm = company.strip().title()
    role_norm = role.strip().title()
    lookup_key = f"{company_norm}|{role_norm}"
    print("Looking up required skills for key:", repr(lookup_key))
    print("Available keys in skill_dict:", list(skill_dict.keys()))

    # Exact match first
    if lookup_key in skill_dict:
        return skill_dict[lookup_key]

    # Fuzzy match
    keys = list(skill_dict.keys())
    matches = get_close_matches(lookup_key, keys, n=1, cutoff=0.7)
    if matches:
        print(f"Fuzzy matched '{lookup_key}' to '{matches[0]}'")
        return skill_dict[matches[0]]

    # Fallback empty
    print(f"No required skills found for '{lookup_key}'")
    return []

# ===============================
# 🔹 6. Post-process Feedback
# ===============================
def post_process_feedback(company, role, candidate_skills, raw_output):
    required_skills = debug_required_skills_lookup(company, role, skill_dict)


    normalize = lambda s: s.strip().lower()
    cand_set = {normalize(s) for s in candidate_skills}
    req_set = {normalize(s) for s in required_skills}

    missing = [s for s in required_skills if normalize(s) not in cand_set]
    match_percent = round(len(req_set & cand_set) / len(req_set) * 100, 2) if req_set else 0

    feedback = (
        f"✅ You already have: {', '.join(candidate_skills)}.\n"
        f"📌 To improve your profile for {role}, focus on: {', '.join(missing) if missing else 'No extra skills needed!'}.\n"
        f"📊 Profile Match: {match_percent}%"
    )

    return {
        "candidate_skills": candidate_skills,
        "required_skills": required_skills,
        "missing_skills": missing,
        "match_percent": match_percent,
        "raw_output": raw_output,
        "final_feedback": feedback
    }

# ===============================
# 🔹 7. FastAPI App
# ===============================
app = FastAPI()

def normalize_string(s: str):
    return s.strip().title() if s else ""

@app.post("/analyze_resume/")
async def analyze_resume(file: UploadFile, company: str = Form(...), role: str = Form(...)):
    try:
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

        raw = generate_feedback(norm_company, norm_role, candidate_skills)
        result = post_process_feedback(norm_company, norm_role, candidate_skills, raw)

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ===============================
# 🔹 8. Run Uvicorn Server
# ===============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
