"""
Root entry point for Depersonalizer Web Service.
Imports and exposes FastAPI application from web.api.
"""

from web.api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
