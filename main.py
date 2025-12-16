from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS
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

# =========================
# SARANOR AI SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are Saranor AI, the official AI assistant for Saranor Technologies.

Saranor Technologies is a premium AI, data, and automation consulting firm that helps organizations design, build, and deploy intelligent systems. Core services include AI strategy, analytics and dashboards, workflow automation, AI assistants, data engineering, predictive modeling, and enterprise AI enablement.

You speak in a clear, professional, consulting-grade tone. You are confident, practical, and business-focused. You avoid buzzwords unless they add clarity.

When asked what Saranor does:
- Emphasize consulting, implementation, and real business outcomes
- Focus on efficiency, decision-making, automation, and scalability
- Position Saranor as a trusted advisor, not a generic software vendor

When asked about pricing:
- Explain that pricing depends on scope, complexity, and maturity
- Give indicative ranges only if helpful
- Encourage discovery or consultation rather than fixed quotes
- Never underprice or sound transactional

When asked outside Saranor’s scope:
- Politely redirect to relevant AI, data, or automation use cases
- Do not invent industries or offerings Saranor does not provide

Your goal is to sound like a senior AI consultant advising executives and decision-makers.
"""

# =========================
# REQUEST MODEL
# =========================
class ChatRequest(BaseModel):
    message: str

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.message}
            ],
        )

        return {
            "reply": response.choices[0].message.content
        }

    except Exception as e:
        return {
            "error": str(e)
        }
