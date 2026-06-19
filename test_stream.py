import cv2

# --- עדכן כאן את הכתובת המדויקת שמופיעה לך בטלפון ---
# שים לב שמשאירים את ה- /video בסוף, זה הנתיב של ה-Stream הגולמי
STREAM_URL = "https://10.100.102.17:8080/video"

def main():
    print(f"[*] Attempting to connect to: {STREAM_URL}")
    
    # פתיחת חיבור הוידאו מול שרת ה-WiFi של הטלפון
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("[ERROR] Could not open video stream. Check your WiFi connection or IP address.")
        return

    print("[SUCCESS] Connected! Press 'q' on your keyboard to exit.")

    while True:
        # קריאת פריים בודד מהמצלמה
        ret, frame = cap.read()
        
        if not ret:
            print("[WARNING] Failed to grab frame.")
            break

        # הצגת הפריים בחלון של OpenCV
        cv2.imshow("Android Live Stream Test", frame)

        # עצירה בלחיצה על מקש q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # שחרור משאבים וסגירת החלון
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Stream closed.")

if __name__ == "__main__":
    main()