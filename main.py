import os
import tempfile
from fastapi import FastAPI, HTTPException, File, UploadFile
import logging

from models import MatchResponse, MatchBreakdown
from services import (
    extract_text_pdf,
    extract_skills,
    calculate_lexical_score,
    calculate_semantic_score,
    evaluate_with_judge
)

app = FastAPI(
    title="CV to JD Matcher API",
    description="A microservice to calculate a hybrid match score between a CV and a Job Description.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/match", response_model=MatchResponse)
async def match_cv_to_jd(
    cv_file: UploadFile = File(..., description="The CV PDF file"),
    jd_file: UploadFile = File(..., description="The Job Description PDF file")
):
    try:
        # Step 0: PDF Text Extraction
        logger.info("Reading uploaded files...")
        
        async def save_and_extract(upload_file: UploadFile) -> str:
            bytes_data = await upload_file.read()
            # We must use suffix='.pdf' so the .endswith('pdf') check passes in extract_text_pdf
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(bytes_data)
                tmp_path = tmp.name
                
            try:
                text = extract_text_pdf(tmp_path)
            finally:
                os.unlink(tmp_path)
            return text
            
        logger.info("Extracting text from PDFs...")
        cv_text = await save_and_extract(cv_file)
        jd_text = await save_and_extract(jd_file)

        # Step 1: LLM Data Extraction
        logger.info("Extracting skills from CV and JD...")
        extracted_skills = extract_skills(cv_text, jd_text)
        
        # Step 2: Lexical Analysis (Jaccard)
        logger.info("Calculating lexical overlap...")
        lexical_score, matched_skills = calculate_lexical_score(
            extracted_skills.cv_skills, 
            extracted_skills.jd_skills
        )
        
        # Step 3: Semantic Similarity
        logger.info("Calculating semantic similarity...")
        semantic_score = calculate_semantic_score(cv_text, jd_text)
        
        # Step 4: LLM-as-a-Judge
        logger.info("Evaluating with LLM as a judge...")
        judge_evaluation = evaluate_with_judge(cv_text, jd_text)
        
        # Step 5: Hybrid Math & Response
        logger.info("Calculating hybrid score...")
        hybrid_score = (0.3 * lexical_score) + (0.3 * semantic_score) + (0.4 * judge_evaluation.judge_score)
        
        breakdown = MatchBreakdown(
            lexical=round(lexical_score, 2),
            semantic=round(semantic_score, 2),
            judge=judge_evaluation.judge_score
        )
        
        response = MatchResponse(
            match_score=round(hybrid_score, 2),
            breakdown=breakdown,
            matched_skills_found=matched_skills,
            skills_to_improve=judge_evaluation.skills_to_improve,
            recommended_projects=judge_evaluation.recommended_projects,
            candidate_feedback=judge_evaluation.candidate_feedback
        )
        
        return response
        
    except ValueError as ve:
        logger.error(f"Validation/Parsing Error: {str(ve)}")
        raise HTTPException(status_code=500, detail=f"LLM Parsing Error: {str(ve)}")
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
