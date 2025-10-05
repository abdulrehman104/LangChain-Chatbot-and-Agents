from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import chat, products

app = FastAPI(
    title="Shop Assistant API",
    description="API for the Shop Assistant application",
    version="1.0.0",
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(products.router)


# uvicorn backend.main:app --> Use this command to run the app