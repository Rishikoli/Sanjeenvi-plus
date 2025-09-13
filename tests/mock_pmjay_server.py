from fastapi import FastAPI, HTTPException, status, Depends, Header, Request
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
import uvicorn
from pydantic import BaseModel
from typing import Optional, Dict, Any
import secrets
import time
import urllib.parse

app = FastAPI()

# Mock database
mock_users = {
    "test_user": {
        "username": "test_user",
        "password": "test_password",
        "client_id": "test_client_id",
        "client_secret": "test_client_secret",
        "claims": {}
    }
}

# Mock token storage
valid_tokens = {}

class TokenData(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class ClaimRequest(BaseModel):
    patient_id: str
    hospital_id: str
    package_code: str
    # Add other claim fields as needed

@app.post("/auth/token")
async def login(request: Request):
    # Manually parse application/x-www-form-urlencoded without python-multipart
    body = await request.body()
    parsed = urllib.parse.parse_qs(body.decode())
    username = (parsed.get("username") or [None])[0]
    password = (parsed.get("password") or [None])[0]
    client_id = (parsed.get("client_id") or [None])[0]
    client_secret = (parsed.get("client_secret") or [None])[0]

    user = mock_users.get(username)
    if not user or user["password"] != password or user["client_id"] != client_id or user["client_secret"] != client_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Generate token
    token = secrets.token_urlsafe(32)
    expires_in = 3600  # 1 hour
    session_id = f"sess_{secrets.token_hex(8)}"
    
    valid_tokens[token] = {
        "username": username,
        "expires_at": time.time() + expires_in,
        "session_id": session_id
    }
    
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "session_id": session_id,
        "refresh_token": f"refresh_{secrets.token_hex(16)}"
    }

async def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token")
    token = authorization.split(" ")[1]
    if token not in valid_tokens or valid_tokens[token]["expires_at"] < time.time():
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    return valid_tokens[token]

@app.post("/claims/submit")
async def submit_claim(request: Request, user = Depends(get_current_user)):
    # Simulate processing delay
    time.sleep(0.1)
    
    # Read JSON body
    payload = await request.json()
    
    # Generate IDs
    submission_id = f"SUB{secrets.token_hex(6).upper()}"
    portal_reference = f"PR-{secrets.token_hex(6).upper()}"
    
    # Store the claim under the user's claims by portal reference
    mock_users[user["username"]]["claims"][portal_reference] = {
        **payload,
        "submission_id": submission_id,
        "portal_reference": portal_reference,
        "status": "submitted",
        "submission_date": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat(),
        "messages": ["Claim received", "Pending review"]
    }
    
    return {
        "submission_id": submission_id,
        "portal_reference": portal_reference,
        "status": "submitted"
    }

@app.get("/claims/{portal_reference}/status")
async def get_claim_status(portal_reference: str, user = Depends(get_current_user)):
    if portal_reference not in mock_users[user["username"]]["claims"]:
        raise HTTPException(status_code=404, detail="Claim not found")
    
    claim = mock_users[user["username"]]["claims"][portal_reference]
    return {
        "portal_reference": portal_reference,
        "status": claim.get("status", "submitted"),
        "submission_date": claim.get("submission_date"),
        "last_updated": claim.get("last_updated"),
        "messages": claim.get("messages", [])
    }

def start_mock_server(port: int = 8000):
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    return server

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
