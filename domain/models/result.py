from pydantic import BaseModel
from .token import Token

class Result(BaseModel):
    predictions : list[Token]