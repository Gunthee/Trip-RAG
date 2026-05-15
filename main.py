import uvicorn
from api import app
from vector_db import init_vector_db

if __name__ == "__main__":
    init_vector_db()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


