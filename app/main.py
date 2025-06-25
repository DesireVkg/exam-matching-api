from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.analyze import analyze_exam
from app.pdf_utils import extract_text_from_pdf

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        path = "temp.pdf"
        with open(path, "wb") as f:
            f.write(contents)
        text = extract_text_from_pdf(path)
        result = analyze_exam(text)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
