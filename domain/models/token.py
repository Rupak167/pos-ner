from pydantic import BaseModel

class Token(BaseModel):
    Token: str
    POS_Tag: str
    NER_Tag: str