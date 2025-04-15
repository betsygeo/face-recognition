from pydantic import BaseModel


class NameFaceRequest(BaseModel):
    name: str