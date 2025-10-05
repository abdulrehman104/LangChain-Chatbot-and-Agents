import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="My FastAPI Application",
    description="This is a sample FastAPI application.",
    version="1.0.0",
)

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uv run .\src\main.py   --> to run the application
# uv run .\src\main.py --reload   --> to run the application with auto-reload