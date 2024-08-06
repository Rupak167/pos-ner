from pydantic import BaseModel, Field
from .token import Token

class Result(BaseModel):
    predictions : list[Token]