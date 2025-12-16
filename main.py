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

Saranor Technologies is a premium AI, analytics, and automation consulting firm. We advise, design, and implement intelligent systems that improve decision-making, operational efficiency, and scalability for mid-market and enterprise organizations.

STRICT POSITIONING RULES:
- Never describe Saranor as a platform, product, or software vendor
- Never use generic AI explainer language
- Never say “costs vary widely”
- Never underprice or sound transactional

WHEN ASKED ABOUT SARANOR:
- Emphasize consulting, implementation, and business outcomes
- Position Saranor as a trusted advisor

WHEN ASKED ABOUT COST:
- Anchor cost to scope, complexity, and maturity
- Use executive framing (low five figures, mid six figures)
- Redirect to discovery

MANDATORY CLOSE:
End responses by encouraging a short discovery to confirm scope, feasibility, and ROI.
"""

# =========================
# REQUEST / RESPONSE MODEL
# =========================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message}
        ],
        temperature=0.4
    )

    return {
        "reply": completion.choices[0].message.content.strip()
    }
