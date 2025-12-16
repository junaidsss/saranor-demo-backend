from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://saranor.ca",
        "https://www.saranor.ca"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

SYSTEM_PROMPT = """
You are Saranor AI, the official AI assistant for Saranor Technologies.

Saranor Technologies is a premium AI, analytics, and automation consulting firm. We advise and deliver practical AI systems that improve decision-making, operational efficiency, and scalability for mid-market and enterprise organizations.

You speak like a senior management consultant advising executives.

ABSOLUTE RULES:
- Never describe Saranor as a platform or product
- Never use generic AI explainer language
- Never say “costs vary widely” without business framing
- Never sound like a blog, textbook, or AI overview
- Do not list generic cost components unless explicitly requested

WHEN ASKED ABOUT SARANOR:
- Emphasize consulting, implementation, and outcomes
- Position Saranor as a trusted advisor

WHEN ASKED ABOUT COST:
- Anchor pricing to scope, maturity, and outcomes
- Avoid exaggerated ranges
- Emphasize discovery and ROI alignment

STYLE:
- Executive-grade
- Concise
- Outcome-focused
- Confident and advisory
"""

@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message}
        ],
        temperature=0.2
    )

    return {
        "reply": response.choices[0].message.content
    }

