import json
import os
from typing import Any, Dict
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class Relevance(BaseModel):
    applies_to_user: bool
    score: float
    reasons: list[str]
    missing_profile_data: list[str]


class ExtractedAction(BaseModel):
    temp_id: str
    action_title: str
    explicit_deadline: str | None = None
    lead_time_days_estimate: float | None = None
    required_prerequisites: list[str]
    source_text_quote: str


class DocumentAnalysisResult(BaseModel):
    category: str = Field(
        description="One of: Exam, Scholarship, Fee, Internship, Hackathon, Event, General Notice, Other"
    )
    action_status: str = Field(
        description="One of: Action Required, Information Only, Optional Opportunity, Already Completed, Not Relevant"
    )
    relevance: Relevance
    extracted_actions: list[ExtractedAction]


def analyze_document(
    raw_document_text: str, user_profile: Dict[str, Any]
) -> DocumentAnalysisResult:
    client = genai.Client()

    system_instruction = (
        "You are the Actra Action Extraction Engine.\n"
        "Your task is to analyze document text, assess applicability against a target profile, "
        "and extract actionable facts without summarizing.\n\n"
        "Strict Rules:\n"
        "- Never hallucinate facts or consequences.\n"
        "- If information is missing, return null or an empty array.\n"
        "- Label every inference clearly.\n"
        "- Determine relevance strictly against the provided profile.\n"
        "- Do not invent deadlines.\n"
        "- Do not invent eligibility.\n"
        "- Do not invent consequences.\n"
        "- Preserve uncertainty when the source is ambiguous."
    )

    prompt = (
        f"TARGET USER PROFILE (JSON):\n{json.dumps(user_profile, indent=2)}\n\n"
        f"DOCUMENT TEXT TO ANALYZE:\n---\n{raw_document_text}\n---"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=DocumentAnalysisResult,
            temperature=0.1,
        ),
    )

    return response.parsed
