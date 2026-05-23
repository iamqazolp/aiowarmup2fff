from pydantic import BaseModel, Field
from typing import List

class MatchBreakdown(BaseModel):
    lexical: float = Field(..., description="Score based on Jaccard similarity of extracted skills (0-100)")
    semantic: float = Field(..., description="Score based on cosine similarity of embeddings (0-100)")
    judge: int = Field(..., description="Score from the LLM-as-a-judge evaluation (0-100)")

class MatchResponse(BaseModel):
    match_score: float = Field(..., description="The final hybrid score (0-100)")
    breakdown: MatchBreakdown
    matched_skills_found: List[str] = Field(default_factory=list, description="Skills found in both CV and JD")
    skills_to_improve: List[str] = Field(default_factory=list, description="Critical skills missing from the CV")
    recommended_projects: List[str] = Field(default_factory=list, description="Actionable projects to bridge skill gaps")
    candidate_feedback: str = Field(..., description="Personalized feedback and advice addressed directly to the candidate")

# Internal schemas for LLM structured output parsing

class ExtractedSkills(BaseModel):
    cv_skills: List[str]
    jd_skills: List[str]

class JudgeEvaluation(BaseModel):
    judge_score: int
    skills_to_improve: List[str]
    recommended_projects: List[str]
    candidate_feedback: str
