import random
from seleniumbase import Driver
import time

# List your profile paths here
PROFILES = [
    "C:/Users/GamingPC/AppData/Local/Google/Chrome/User Data/Default",
    "C:/Users/GamingPC/AppData/Local/Google/Chrome/User Data/Profile 1"
]

def vehicle_scraper_final_test(reg_no):
    # Select a random profile to distribute the load
    selected_profile = random.choice(PROFILES)

    driver = Driver(uc=True, user_data_dir=selected_profile, headless=False)
    
    try:
        driver.get("https://excise.gos.pk/vehicle/vehicle_search")
        
        # 1. Enter Registration Number
        driver.wait_for_element("input#reg_no", timeout=20)
        driver.type("input#reg_no", reg_no)

        # 2. Handle Captcha
        driver.switch_to_frame('iframe[title="reCAPTCHA"]')
        driver.uc_click("span#recaptcha-anchor")
        driver.switch_to_default_window()

        # 3. Wait for Token
        verified = False
        for _ in range(60):
            token = driver.get_attribute('textarea#g-recaptcha-response', 'value')
            if token and len(token) > 20:
                verified = True
                break
            time.sleep(1)

        if not verified:
            print("Captcha timeout.")
            return

        # 4. FORCE THE CLICK
        print("Clicking Search (Forced JS Click)...")
        # Small wait for the site's JS to catch up
        time.sleep(2) 
        # Using js_click bypasses the 'disabled' attribute issue
        driver.js_click("button#search_veh")
        
        # 5. Data Extraction
        time.sleep(5) 
        cells = driver.find_elements("css selector", "table td")
        
        if not cells:
            print("No results. Check if the page actually loaded the table.")
        else:
            print("\n" + "="*40)
            for index, cell in enumerate(cells):
                print(f"Index {index}: {cell.text.strip()}")
            print("="*40)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    vehicle_scraper_final_test("BFF-029")