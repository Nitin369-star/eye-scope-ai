from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import FileResponse
from utils import process_batch
import shutil
import tempfile
import zipfile
import os

router = APIRouter()

@router.post("/batch")
async def batch_predict(
    files: list[UploadFile],
    patient_name: str = Form(...),
    patient_age: str = Form(...),
    phone: str = Form(""),
    email: str = Form(""),
    language: str = Form("English"),
    generate_pdfs: bool = Form(False)
):
    results, pdf_paths = process_batch(files, patient_name, patient_age, phone, email, language, generate_pdfs)

    if generate_pdfs and pdf_paths:
        zip_filename = f"{patient_name.replace(' ', '_')}_reports.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for pdf_path in pdf_paths:
                zipf.write(pdf_path, os.path.basename(pdf_path))
                os.remove(pdf_path)
        return FileResponse(zip_filename, media_type="application/zip", filename=zip_filename)

    return {"status": "success", "results": results}
