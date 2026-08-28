import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR

app = FastAPI(title="PaddleOCR Service", version="2.0.0")

_ocr: PaddleOCR | None = None
_executor = ThreadPoolExecutor(max_workers=2)


def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )
    return _ocr


@app.on_event("startup")
async def _warmup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, get_ocr)


def _result_to_dict(res, page_num: int) -> dict[str, Any]:
    lines = []
    full_text_parts = []

    dt_polys = res.get("dt_polys", []) or []
    rec_texts = res.get("rec_texts", []) or []
    rec_scores = res.get("rec_scores", []) or []

    for poly, text, score in zip(dt_polys, rec_texts, rec_scores):
        lines.append({
            "text": text,
            "confidence": round(float(score), 4),
            "box": {
                "top_left":     {"x": float(poly[0][0]), "y": float(poly[0][1])},
                "top_right":    {"x": float(poly[1][0]), "y": float(poly[1][1])},
                "bottom_right": {"x": float(poly[2][0]), "y": float(poly[2][1])},
                "bottom_left":  {"x": float(poly[3][0]), "y": float(poly[3][1])},
            },
        })
        full_text_parts.append(text)

    return {
        "page": page_num,
        "full_text": "\n".join(full_text_parts),
        "lines": lines,
    }


def _run_ocr_on_file(file_path: str, filename: str) -> dict[str, Any]:
    results = list(get_ocr().predict(file_path))

    pages = []
    for i, res in enumerate(results, start=1):
        pages.append(_result_to_dict(res, i))

    return {
        "filename": filename,
        "total_pages": len(pages),
        "pages": pages,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    loop = asyncio.get_event_loop()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await loop.run_in_executor(
            _executor,
            partial(_run_ocr_on_file, tmp_path, file.filename),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"OCR failed: {exc}")
    finally:
        os.unlink(tmp_path)

    return JSONResponse(result)


@app.post("/ocr/image")
async def ocr_image(file: UploadFile = File(...)):
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {ext}")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    loop = asyncio.get_event_loop()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await loop.run_in_executor(
            _executor,
            partial(_run_ocr_on_file, tmp_path, file.filename),
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"OCR failed: {exc}")
    finally:
        os.unlink(tmp_path)

    page = result["pages"][0] if result["pages"] else {"page": 1, "full_text": "", "lines": []}
    return JSONResponse(page)
