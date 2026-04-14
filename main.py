import threading
from fastapi import FastAPI, HTTPException
import mysql.connector
import random
from seleniumbase import Driver
import time
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
from fastapi.responses import StreamingResponse
from vision_pipeline import ANPRPipeline
import numpy as np # Just in case
import os 

app = FastAPI()

# Add this block to fix the connection error
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],
)

# Configuration
PROFILES = [
    "C:/Users/GamingPC/AppData/Local/Google/Chrome/User Data/Default",
    "C:/Users/GamingPC/AppData/Local/Google/Chrome/User Data/Profile 1"
]

def get_db_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="", database="fyp"
    )

def split_vehicle_details(details_str):
    parts = details_str.strip().rsplit(' ', 1)
    
    if len(parts) == 2:
        make_model = parts[0]
        year = parts[1]
        brand_parts = make_model.split(' ', 1)
        make = brand_parts[0] if len(brand_parts) > 0 else "Unknown"
        model = brand_parts[1] if len(brand_parts) > 1 else "Unknown"
        
        return make, model, year
    return details_str, "Unknown", "Unknown"
# --- 3. UPDATED AI WORKER (With Debouncing) ---
def ai_vision_worker():
    print("AI Vision Worker: Starting up...")
    from vision_pipeline import ANPRPipeline
    pipeline = ANPRPipeline()
    
    scan_count = 0
    
    while True:
        try:
            scan_count += 1
            print(f"\n--- AI Scan Iteration {scan_count} ---")
            
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            cursor.execute("""
                SELECT b.id, b.bay_name, b.x_min, b.y_min, b.x_max, b.y_max, c.stream_index
                FROM parking_bays b
                JOIN cameras c ON b.camera_id = c.id
            """)
            bays = cursor.fetchall()
            update_cursor = db.cursor()
            
            for bay in bays:
                bay_id = bay['id']
                bay_name = bay['bay_name']
                stream_idx = str(bay['stream_index'])
                
                if stream_idx in LATEST_FRAMES:
                    frame = LATEST_FRAMES[stream_idx]
                    
                    # 1. Validate Crop Coordinates
                    y1, y2 = bay['y_min'], bay['y_max']
                    x1, x2 = bay['x_min'], bay['x_max']
                    bay_crop = frame[y1:y2, x1:x2]
                    
                    if bay_crop.size == 0:
                        print(f"  [{bay_name}] WARNING: Invalid crop coordinates! ({x1},{y1}) to ({x2},{y2})")
                        continue
                    
                    # DEBUG FEATURE: Save the crop so you can see exactly what YOLO sees
                    cv2.imwrite(f"debug_crop_{bay_name}.jpg", bay_crop)
                    
                    # 2. Pipeline Prediction
                    detected_plate = pipeline.process_bay_image(bay_crop)
                    print(f"  [{bay_name}] Raw Pipeline Output: '{detected_plate}'")
                    
                    # 3. DEBOUNCING LOGIC
                    state = BAY_STATES.get(bay_id, {"current_raw": None, "streak": 0, "confirmed": "INIT"})
                    
                    if state["current_raw"] == detected_plate:
                        state["streak"] += 1
                    else:
                        state["current_raw"] = detected_plate
                        state["streak"] = 1
                        
                    BAY_STATES[bay_id] = state
                    print(f"  [{bay_name}] Debounce Tracker -> Current: {state['current_raw']}, Streak: {state['streak']}/{DEBOUNCE_LIMIT}")
                    
                    # 4. Database Update
                    if state["streak"] >= DEBOUNCE_LIMIT and state["confirmed"] != detected_plate:
                        state["confirmed"] = detected_plate
                        print(f"  >>> [{bay_name}] CONFIRMED State Change -> {detected_plate} <<<")
                        
                        if detected_plate:
                            update_cursor.execute("""
                                UPDATE parking_bays SET current_status = 'occupied', last_plate_detected = %s WHERE id = %s
                            """, (detected_plate, bay_id))
                        else:
                            update_cursor.execute("""
                                UPDATE parking_bays SET current_status = 'vacant' WHERE id = %s
                            """, (bay_id,))
                        db.commit()
                else:
                    print(f"  [{bay_name}] ERROR: No frame available in memory for camera index '{stream_idx}'.")

            db.close()
        except Exception as e:
            print(f"AI Vision Worker Error: {e}")
            
        time.sleep(3) # Wait 3 seconds between facility scans


def background_worker():
    print("Background Worker: Thread started successfully.")
    while True:
        try:
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            
            # Find plates in bays that aren't in our cache
            query = """
                SELECT DISTINCT b.last_plate_detected 
                FROM parking_bays b
                LEFT JOIN vehicle_summary v ON b.last_plate_detected = v.registration_no
                WHERE b.current_status = 'occupied' 
                AND b.last_plate_detected IS NOT NULL
                AND v.registration_no IS NULL
            """
            cursor.execute(query)
            missing_plates = cursor.fetchall()
            
            if not missing_plates:
                # This print tells us the thread IS alive, just has no work
                print(f"Background Worker: Checked at {time.strftime('%H:%M:%S')} - No unidentified plates found.")
            else:
                print(f"Background Worker: Found {len(missing_plates)} plates to identify: {[p['last_plate_detected'] for p in missing_plates]}")
                for row in missing_plates:
                    plate = row['last_plate_detected']
                    print(f"Background Worker: Launching browser for {plate}...")
                    
                    try:
                        scraped = scrape_vehicle(plate)
                        if scraped:
                            make, model, year = split_vehicle_details(scraped['details'])
                            
                            save_query = """
                                INSERT INTO vehicle_summary 
                                (registration_no, make, vehicle_model, model_year, engine_no) 
                                VALUES (%s, %s, %s, %s, %s)
                            """
                            save_cursor = db.cursor()
                            save_cursor.execute(save_query, (
                                scraped['registration_no'], make, model, year, scraped['engine_no']
                            ))
                            db.commit()
                            print(f"Background Worker: Database updated for {plate}")
                        else:
                            print(f"Background Worker: Scrape failed (No data) for {plate}")
                    except Exception as e:
                        print(f"Background Worker: Scrape Error for {plate}: {e}")
            
            db.close()
        except Exception as e:
            print(f"Background Worker: Database Connection Error: {e}")
        
        # Reduced to 30 seconds for testing
        time.sleep(30)

def scrape_vehicle(reg_no):
    selected_profile = random.choice(PROFILES)
    try:
        driver = Driver(
            uc=True, 
            user_data_dir=selected_profile, 
            headless=False,
            incognito=False
        )
    except Exception as e:
        print(f"Driver Init Failed: {e}")
        return None
    
    try:
        driver.set_page_load_timeout(30)
        driver.get("https://excise.gos.pk/vehicle/vehicle_search")
        driver.wait_for_element("input#reg_no", timeout=20)
        driver.type("input#reg_no", reg_no)

        # Captcha logic
        driver.switch_to_frame('iframe[title="reCAPTCHA"]')
        driver.uc_click("span#recaptcha-anchor")
        driver.switch_to_default_window()

        # Wait for Token
        for _ in range(60):
            token = driver.get_attribute('textarea#g-recaptcha-response', 'value')
            if token and len(token) > 20:
                break
            time.sleep(1)

        # Forced JS Click
        time.sleep(2)
        driver.js_click("button#search_veh")
        
        # Extraction based on your verified indices
        time.sleep(5)
        cells = driver.find_elements("css selector", "table td")
        
        if len(cells) > 6:
            return {
                "registration_no": cells[4].text.strip(),
                "details": cells[5].text.strip(), # This has Make, Model, Year
                "engine_no": cells[6].text.strip()
            }
        return None
    finally:
        driver.quit()

# --- SHARED MEMORY & STATE ---
LATEST_FRAMES = {} # { "stream_index": numpy_array }
BAY_STATES = {}    # { bay_id: {"current_raw": "BFF-029", "streak": 0, "confirmed": "BFF-029"} }
DEBOUNCE_LIMIT = 3 # Needs to see the same thing 3 times in a row to trust it

def camera_manager_worker():
    """Runs continuously in the background, keeping all camera feeds fresh."""
    print("Camera Manager: Starting up...")
    caps = {} # Store active VideoCapture objects
    
    while True:
        try:
            db = get_db_connection()
            cursor = db.cursor(dictionary=True)
            cursor.execute("SELECT stream_index FROM cameras")
            cameras = cursor.fetchall()
            db.close()
            
            active_streams = [str(c['stream_index']) for c in cameras]
            
            # Close deleted cameras
            for stream in list(caps.keys()):
                if stream not in active_streams:
                    caps[stream].release()
                    del caps[stream]
                    if stream in LATEST_FRAMES: del LATEST_FRAMES[stream]
            
            # Open new cameras & read frames
            for stream in active_streams:
                if stream not in caps:
                    idx = int(stream) if stream.isdigit() else stream
                    caps[stream] = cv2.VideoCapture(idx)
                    caps[stream].set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    caps[stream].set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                ret, frame = caps[stream].read()
                if ret:
                    LATEST_FRAMES[stream] = frame.copy()
                    
        except Exception as e:
            print(f"Camera Manager Error: {e}")
            
        time.sleep(0.1) # Fetch frames at roughly 10 FPS to save CPU

# --- 2. UPDATED VIDEO FEED (Reads from memory, not directly from camera) ---
def generate_frames(stream_index):
    stream_index = str(stream_index)
    while True:
        if stream_index in LATEST_FRAMES:
            frame = LATEST_FRAMES[stream_index]
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

@app.get("/api/video_feed/{stream_index}")
def video_feed(stream_index: str):
    return StreamingResponse(generate_frames(stream_index), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/dashboard/status")
def get_dashboard_status():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    # We join the bays with the vehicle details if a plate is present
    query = """
        SELECT 
            b.id, b.bay_name, b.current_status, b.last_plate_detected,
            v.make, v.vehicle_model, v.model_year
        FROM parking_bays b
        LEFT JOIN vehicle_summary v ON b.last_plate_detected = v.registration_no
        ORDER BY b.bay_name ASC
    """
    cursor.execute(query)
    bays = cursor.fetchall()
    
    # Calculate quick stats for the header cards
    total = len(bays)
    occupied = sum(1 for b in bays if b['current_status'] == 'occupied')
    vacant = total - occupied
    
    db.close()
    return {
        "stats": {"total": total, "occupied": occupied, "vacant": vacant},
        "bays": bays
    }

@app.get("/api/vehicle/{reg_no}")
def get_vehicle(reg_no: str):
    reg_no = reg_no.upper().strip()
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    # 1. Check MySQL Cache
    cursor.execute("SELECT * FROM vehicle_summary WHERE registration_no = %s", (reg_no,))
    cache = cursor.fetchone()
    
    if cache:
        db.close()
        return {"status": "success", "source": "cache", "data": cache}

    # 2. Scrape if not found
    scraped = scrape_vehicle(reg_no)
    
    if scraped:
        # Split the "details" string into Make, Model, and Year
        make, model, year = split_vehicle_details(scraped['details'])
        
        # 3. Save to MySQL with separate columns
        sql = """
        INSERT INTO vehicle_summary 
        (registration_no, make, vehicle_model, model_year, engine_no) 
        VALUES (%s, %s, %s, %s, %s)
        """
        values = (
            scraped['registration_no'], 
            make, 
            model, 
            year, 
            scraped['engine_no']
        )
        cursor.execute(sql, values)
        db.commit()
        db.close()
        
        return {
            "status": "success", 
            "source": "live", 
            "data": {
                "registration_no": scraped['registration_no'],
                "make": make,
                "model": model,
                "year": year,
                "engine_no": scraped['engine_no']
            }
        }
    
    db.close()
    raise HTTPException(status_code=404, detail="Vehicle not found")

# --- DATA MODELS ---
class CameraCreate(BaseModel):
    camera_name: str
    stream_index: str

class BayCreate(BaseModel):
    camera_id: int # NEW: Link to camera
    bay_name: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int





# --- CAMERA AND BAY MANAGEMENT ENDPOINTS ---
@app.get("/api/cameras")
def get_cameras():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM cameras")
    cameras = cursor.fetchall()
    db.close()
    return {"status": "success", "cameras": cameras}

@app.get("/api/bays")
def get_all_bays():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, camera_id, bay_name, x_min, y_min, x_max, y_max FROM parking_bays")
    bays = cursor.fetchall()
    db.close()
    return {"status": "success", "bays": bays}

@app.post("/api/bays")
def create_bay(bay: BayCreate):
    db = get_db_connection()
    cursor = db.cursor()
    sql = """
        INSERT INTO parking_bays (camera_id, bay_name, x_min, y_min, x_max, y_max, current_status) 
        VALUES (%s, %s, %s, %s, %s, %s, 'vacant')
    """
    cursor.execute(sql, (bay.camera_id, bay.bay_name, bay.x_min, bay.y_min, bay.x_max, bay.y_max))
    db.commit()
    db.close()
    return {"status": "success"}

@app.post("/api/cameras")
def add_camera(cam: CameraCreate):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("INSERT INTO cameras (camera_name, stream_index) VALUES (%s, %s)", (cam.camera_name, cam.stream_index))
    db.commit()
    db.close()
    return {"status": "success"}

@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: int):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("DELETE FROM cameras WHERE id = %s", (camera_id,))
    db.commit()
    db.close()
    return {"status": "success", "message": "Camera deleted"}

@app.delete("/api/bays/{bay_id}")
def delete_bay(bay_id: int):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("DELETE FROM parking_bays WHERE id = %s", (bay_id,))
    db.commit()
    db.close()
    return {"status": "success", "message": "Bay deleted"}

threading.Thread(target=camera_manager_worker, daemon=True).start()
threading.Thread(target=ai_vision_worker, daemon=True).start()
threading.Thread(target=background_worker, daemon=True).start() # Your scraper worker

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)