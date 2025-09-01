from pydantic import BaseModel, Field
from typing import Optional


class CountFilesInput(BaseModel):
    directory_path: str = Field(..., description="Path to the directory")


class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file")


class CreateNoteInput(BaseModel):
    note_content: str = Field(..., description="Content of the note")
    note_name: Optional[str] = Field(None, description="Name of the note file")


class FetchCompanyInfoInput(BaseModel):
    inn: str = Field(..., description="ИНН компании (10 или 12 цифр)")


class PromptRequest(BaseModel):
    prompt: str
    session_id: str = None
