import urllib.request
import json

# --- עדכן ל-IP הנוכחי של הטלפון שלך ---
SENSOR_URL = "http://10.100.102.17:8080/sensors.json"

def debug_sensors():
    print(f"[*] Fetching sensor data from: {SENSOR_URL}")
    try:
        with urllib.request.urlopen(SENSOR_URL, timeout=2.0) as url:
            raw_data = url.read().decode()
            data = json.loads(raw_data)
            
            print("\n[✅] Successfully connected to sensors endpoint!")
            print(f"Available top-level keys in JSON: {list(data.keys())}\n")
            
            # בודק ומדפיס דוגמה למפתחות הג'יירו הנפוצים
            for key in ['GYRx', 'GYRy', 'GYRz', 'gyro']:
                if key in data:
                    print(f"=== Found Key: '{key}' ===")
                    # הדפסת הנתונים הגולמיים של החיישן הספציפי כדי לראות את המבנה
                    print(json.dumps(data[key], indent=2)[:500]) # מדפיס רק את ההתחלה שלא יציף
                    print("-" * 30)
                else:
                    print(f"❌ Key '{key}' not found in JSON.")
                    
    except Exception as e:
        print(f"\n[ERROR] Failed to read or parse sensors: {e}")

if __name__ == "__main__":
    debug_sensors()