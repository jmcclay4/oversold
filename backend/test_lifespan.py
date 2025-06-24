from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan function triggered")
    print("Server starting up...")
    yield
    print("Server shutting down...")

print("Registering lifespan function")
app.lifespan = lifespan

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    print("Running Uvicorn server...")
    uvicorn.run("test_lifespan:app", host="0.0.0.0", port=8000, reload=True)