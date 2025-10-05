from typing import List
from logger import logger
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File
from modules.load_vectorstore import load_vectorstore


router=APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile] = File(...)):
    try:
        logger.info("Recieved uploaded files")
        load_vectorstore(files)
        
        logger.info("Document added to vectorstore")
        return {"messages":"Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500,content={"error":str(e)})
