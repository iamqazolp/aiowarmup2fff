import fitz
import os
import json
from typing import Tuple, List
from sentence_transformers import SentenceTransformer
import numpy as np
from groq import Groq
from dotenv import load_dotenv

from models import ExtractedSkills, JudgeEvaluation

# Load environment variables
load_dotenv()

# Initialize SentenceTransformer globally so it only loads once on startup
print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

# Initialize Groq client
# This assumes GROQ_API_KEY is set in the environment or .env file
groq_client = Groq()
GROQ_MODEL = "llama-3.3-70b-versatile"

def extract_text_pdf(file_path: str):
    if not file_path.endswith('pdf'):
        raise Exception('File extension must be pdf!')
    pdf = fitz.open(file_path)
    pdf_texts = [page.get_text('text') for page in pdf]
    pdf.close()
    return ' '.join(pdf_texts)

def extract_skills(cv_text: str, jd_text: str) -> ExtractedSkills:
    prompt = f"""
    You are an expert technical recruiter. Extract the technical skills from the following CV and Job Description (JD).
    
    CRITICAL RULES FOR CV EXTRACTION:
    - ONLY extract skills that the candidate actually possesses or has experience with.
    - DO NOT extract skills that the candidate explicitly states they DO NOT know, lack experience with, or are currently learning but do not know yet.
    
    Normalize all skills to lowercase (e.g., "Python", "PYTHON", and "python" all become "python").
    
    Return a strict JSON object with exactly two keys:
    - "cv_skills": a list of strings representing the skills possessed by the candidate in the CV.
    - "jd_skills": a list of strings representing the skills required or preferred in the JD.

    CV Text:
    {cv_text}

    JD Text:
    {jd_text}
    """
    
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        model=GROQ_MODEL,
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    try:
        content = response.choices[0].message.content
        data = json.loads(content)
        # Ensure lowercase (though prompted, it's safer to enforce in code)
        data['cv_skills'] = [s.lower() for s in data.get('cv_skills', [])]
        data['jd_skills'] = [s.lower() for s in data.get('jd_skills', [])]
        return ExtractedSkills(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing structured output: {str(e)}")

def calculate_lexical_score(cv_skills: List[str], jd_skills: List[str]) -> Tuple[float, List[str]]:
    if not jd_skills:
        return 0.0, []
        
    set_cv = set(cv_skills)
    set_jd = set(jd_skills)
    
    intersection = set_cv.intersection(set_jd)
    overlap_percentage = (len(intersection) / len(set_jd)) * 100.0
    
    # Cap at 100
    overlap_percentage = min(overlap_percentage, 100.0)
    
    return overlap_percentage, list(intersection)

def calculate_semantic_score(cv_text: str, jd_text: str) -> float:
    # Compute embeddings
    embeddings = embedding_model.encode([cv_text, jd_text])
    
    cv_emb = embeddings[0]
    jd_emb = embeddings[1]
    
    # Compute cosine similarity
    dot_product = np.dot(cv_emb, jd_emb)
    norm_cv = np.linalg.norm(cv_emb)
    norm_jd = np.linalg.norm(jd_emb)
    
    if norm_cv == 0 or norm_jd == 0:
        return 0.0
        
    cos_sim = dot_product / (norm_cv * norm_jd)
    
    # Scale to [0, 100], assuming cos_sim is largely between 0 and 1 for text
    scaled_sim = max(0.0, float(cos_sim)) * 100.0
    
    return min(scaled_sim, 100.0)

def evaluate_with_judge(cv_text: str, jd_text: str) -> JudgeEvaluation:
    prompt = f"""
    Act as a senior technical recruiter. Evaluate the candidate's CV against the Job Description.
    Provide a holistic match score out of 100, identify critical skills missing from the CV, 
    recommend 2 specific, actionable portfolio projects that would bridge the skill gaps,
    and write personalized feedback addressed directly to the candidate.

    Return a strict JSON object with exactly these keys:
    - "judge_score": an integer between 0 and 100.
    - "skills_to_improve": a list of strings detailing missing critical skills.
    - "recommended_projects": a list of exactly 2 strings, each describing a specific portfolio project.
    - "candidate_feedback": a string containing personalized, constructive feedback addressed to the candidate ("You have strong experience in... but you should focus on...").

    CV Text:
    {cv_text}

    JD Text:
    {jd_text}
    """
    
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        model=GROQ_MODEL,
        response_format={"type": "json_object"},
        temperature=0.2
    )
    
    try:
        content = response.choices[0].message.content
        data = json.loads(content)
        return JudgeEvaluation(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM judge response as JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error parsing judge output: {str(e)}")
