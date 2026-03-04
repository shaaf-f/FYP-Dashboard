import threading
from fastapi import FastAPI, HTTPException
import mysql.connector
import random
from seleniumbase import Driver
import time
from fastapi.middleware.cors import CORSMiddleware

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
        time.sleep(10)

def scrape_vehicle(reg_no):
    selected_profile = random.choice(PROFILES)
    try:
        driver = Driver(
            uc=True, 
            user_data_dir=selected_profile, 
            headless=True,
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

threading.Thread(target=background_worker, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)