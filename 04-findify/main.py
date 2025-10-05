from fastapi import FastAPI

app = FastAPI(
    title="Findify API",
    description="A simple API for Findify",
    version="1.0.0",
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Findify API!"}


def main():
    print("Hello from findify!")


if __name__ == "__main__":
    main()
