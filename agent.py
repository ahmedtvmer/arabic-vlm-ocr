import base64
import json
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class LegalArticle(BaseModel):
    article_number: str
    content: str

class OCRResult(BaseModel):
    document_type: Literal["official_letter", "regulation", "report", "unknown"]
    issuing_authority: Optional[str] = None
    full_text: str
    legal_articles: List[LegalArticle] = Field(default_factory=list)
    confidence_score: float

class AgentState(TypedDict):
    image_path: str
    raw_ocr_json: Optional[str]
    validated_data: Optional[OCRResult]
    final_report: str

def extraction_node(state: AgentState):
    """Pings the local llama-server (GGUF) for raw OCR."""
    print(" [Step 1] Local VLM: Extracting structured data...")
    
    with open(state["image_path"], "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
    
    response = client.chat.completions.create(
        model="qwen2-vl-arabic-ocr-merged",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract OCR as JSON. Use fields: document_type, issuing_authority, full_text, legal_articles."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        temperature=0.0
    )
    return {"raw_ocr_json": response.choices[0].message.content}

def validation_reasoning_node(state: AgentState):
    """Uses a high-tier LLM to validate the local output and clean up hallucinations."""
    print(" [Step 2] Reasoning Node: Validating and cleaning data...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    structured_llm = llm.with_structured_output(OCRResult)
    
    system_instruction = (
        "You are an Arabic legal expert. You will receive raw OCR output. "
        "Your job is to: 1. Correct spelling mistakes in the Arabic text. "
        "2. Remove any obvious hallucinations (e.g., mentions of Saudi if not present). "
        "3. Map the content to the requested JSON schema."
    )
    
    cleaned_data = structured_llm.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=f"Raw OCR Data: {state['raw_ocr_json']}")
    ])
    
    return {
        "validated_data": cleaned_data,
        "final_report": f"Successfully processed {cleaned_data.document_type}. Confidence: {cleaned_data.confidence_score}"
    }

workflow = StateGraph(AgentState)

workflow.add_node("extract", extraction_node)
workflow.add_node("validate", validation_reasoning_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", END)

app = workflow.compile()

if __name__ == "__main__":
    # Ensure llama-server is running on :8080
    config = {"image_path": "./test_data/page_002.jpg"}
    
    result = app.invoke(config)
    
    print("\n--- FINAL STRUCTURED OUTPUT ---")
    print(json.dumps(result["validated_data"].dict(), indent=2, ensure_ascii=False))
    print(f"\nStatus: {result['final_report']}")