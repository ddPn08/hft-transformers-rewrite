from pydantic import BaseModel

from modules.transcriber import TranscriberConfig

class Config(BaseModel):
    model: TranscriberConfig

