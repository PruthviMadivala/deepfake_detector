from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import json
from hashlib import sha256

from hf_model import load_hf_model, predict_hf
from gradcam import generate_gradcam
from report_gen import save_last_result

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================
# USER DATABASE (JSON FILE)
# ====================================================
USERS_FILE = Path("users.json")

if not USERS_FILE.exists():
    USERS_FILE.write_text(json.dumps({"users": []}, indent=4))


# Helper to read user DB
def load_users():
    return json.loads(USERS_FILE.read_text())


# Helper to save user DB
def save_users(data):
    USERS_FILE.write_text(json.dumps(data, indent=4))


# ====================================================
# REGISTER API
# ====================================================
@app.post("/register")
async def register(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    users = load_users()

    # Check if email already exists
    for u in users["users"]:
        if u["email"] == email:
            return {"success": False, "message": "Email already registered"}

    hashed = sha256(password.encode()).hexdigest()

    users["users"].append({
        "name": name,
        "email": email,
        "password": hashed
    })

    save_users(users)

    return {"success": True, "message": "Registered successfully"}


# ====================================================
# LOGIN API
# ====================================================
@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    users = load_users()
    hashed = sha256(password.encode()).hexdigest()

    for u in users["users"]:
        if u["email"] == email and u["password"] == hashed:
            return {
                "success": True,
                "message": "login ok",
                "user": {
                    "name": u["name"],
                    "email": u["email"]
                }
            }

    return {"success": False, "message": "Invalid credentials"}


# ====================================================
# LOAD MODEL ONCE
# ====================================================
model = load_hf_model()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ====================================================
# PREDICT API
# ====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_hf(model, str(file_path))
    save_last_result(result)
    return result


# ====================================================
# GRADCAM
# ====================================================
@app.post("/gradcam")
async def gradcam_route(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    heatmap = generate_gradcam(model, str(file_path))
    return {"heatmap": heatmap}


# ====================================================
# REPORT
# ====================================================
@app.get("/report")
async def get_report():
    last = Path("last_result.json")
    if last.exists():
        return last.read_text()
    return {"error": "No report available"}
