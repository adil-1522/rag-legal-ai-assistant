from pydantic import BaseModel,Field
from typing import List,Optional
import json

class Party(BaseModel):
    name:str=Field(description="Name of the part involved")
    role:str=Field(description="Role e.g. Buyer,Seller,Employer,Employee")

class Risk(BaseModel):
    description:str=Field(description="Description of the risk or liability")
    severity:str=Field(description="Severity:High,Medium,Low")

class KeyDate(BaseModel):
    event:str=Field(description="What happens on this date")
    date:str=Field(description="The actual date or deadline")

class PaymentTerm(BaseModel):
    description:str=Field(description="Description of the payment term")
    amount:Optional[str]=Field(default=None,description="Amount if mentioned")

class ContractAnalysis(BaseModel):
    """
    JSON Schema for full contract analysis.
    Pydantic auto-generates the schema — LLM must follow it.
    """
    summary: str = Field(description="2-3 sentence summary of the contract")
    parties: List[Party] = Field(description="All parties involved")
    risks: List[Risk] = Field(description="All risks and liabilities")
    key_dates: List[KeyDate] = Field(description="All important dates and deadlines")
    payment_terms: List[PaymentTerm] = Field(description="All payment terms")
    obligations: List[str] = Field(description="Key obligations of each party")
    termination_conditions: List[str] = Field(description="Termination conditions")
    jurisdiction: Optional[str] = Field(default=None, description="Governing law")
    penalty_clauses: List[str] = Field(description="Any penalty or breach clauses") 
    
    @classmethod
    def get_json_schema(cls) -> str:
        """Returns the JSON schema string to inject into prompts"""
        return json.dumps(cls.model_json_schema(), indent=2)
    
class QuestionAnswer(BaseModel):
    """
    JSON Schema for Q&A structured output.
    """
    answer: str = Field(description="Direct answer to the question")
    relevant_clauses: List[str] = Field(description="Relevant clauses from contract")
    confidence: str = Field(description="Confidence level: High, Medium, or Low")
    follow_up_suggestions: List[str] = Field(description="2-3 follow-up questions")

    @classmethod
    def get_json_schema(cls) -> str:
        """Returns the JSON schema string to inject into prompts"""
        return json.dumps(cls.model_json_schema(), indent=2)