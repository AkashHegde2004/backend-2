from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class TeacherRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class TeacherLogin(BaseModel):
    email: EmailStr
    password: str

class StudentCreate(BaseModel):
    name: str
    usn: str
    semester: str
    subject: str
    department: str

class StudentResponse(BaseModel):
    id: str
    name: str
    usn: str
    semester: str
    subject: str
    department: str
    created_at: str

class SessionCreate(BaseModel):
    subject: str
    semester: str
    department: str

class SessionResponse(BaseModel):
    id: str
    teacher_id: str
    subject: str
    semester: str
    department: str
    start_time: str
    end_time: Optional[str] = None
    status: str

class AttendanceRecord(BaseModel):
    student_usn: str
    student_name: str
    timestamp: str
    attentiveness_scores: List[dict]  # [{time: str, label: str}]

class AttendanceResponse(BaseModel):
    session_id: str
    records: List[AttendanceRecord]