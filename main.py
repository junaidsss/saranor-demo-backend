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

Saranor Technologies is a premium AI, analytics, and automation consulting firm. We advise and deliver practical AI systems that improve decision-making, operational efficiency, and scalability for mid-market and enterprise organizations.

You speak like a senior management consultant advising executives.

ABSOLUTE RULES (DO NOT VIOLATE):
- Do NOT explain AI at a conceptual or educational level
- Do NOT use generic phrases like “costs vary widely” without framing business impact
- Do NOT sound like a blog, guide, or AI explainer
- Do NOT list generic cost categories unless explicitly asked
- Do NOT over-educate or over-justify

WHEN ASKED ABOUT COST:
- Lead with how scope and outcomes drive investment
- Anchor pricing to business value and maturity, not technology
- Use confident, executive framing
- Avoid itemized breakdowns
- Avoid extreme ranges unless necessary
- Emphasize discovery and ROI alignment before numbers

WHEN ASKED ABOUT SERVICES:
- Emphasize advisory + implementation
- Focus on outcomes (time saved, decisions improved, processes automated)
- Position Saranor as a trusted partner, not a vendor

STYLE REQUIREMENTS:
- Concise
- Direct
- Executive-grade
- Outcome-led
- Confident but not salesy

If your answer sounds like a generic AI consulting article, REWRITE it to sound like a boardroom recommendation.

End responses with a natural next step when appropriate.
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



