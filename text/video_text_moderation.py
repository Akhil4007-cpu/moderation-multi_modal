import cv2
import pytesseract
import subprocess
import speech_recognition as sr
import re
import os

# ----------------------------
# 0. Setup
# ----------------------------
# Tesseract path (update if different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Unsafe keywords (expand as needed)
UNSAFE_KEYWORDS = [
    "suicide", "murder", "kill", "drugs", "weapon", "crime",
    "adult", "abuse", "fight", "fuck", "bitch", "bloody", "asshole"
]

# ----------------------------
# 1. Extract text from video frames (OCR)
# ----------------------------
def extract_text_from_frames(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    extracted_texts = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if text.strip():
                extracted_texts.append(text.strip())

        frame_count += 1

    cap.release()
    return " ".join(extracted_texts) if extracted_texts else ""

# ----------------------------
# 2. Extract audio → text (SpeechRecognition)
# ----------------------------
def extract_text_from_audio(video_path):
    audio_file = "temp_audio.wav"
    # Extract audio using ffmpeg
    command = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_file}"'
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    recognizer = sr.Recognizer()
    audio_text = ""
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            audio_text = recognizer.recognize_google(audio)
    except:
        pass

    if os.path.exists(audio_file):
        os.remove(audio_file)

    return audio_text.strip()

# ----------------------------
# 3. Find unsafe words
# ----------------------------
def find_unsafe_words(text):
    found = []
    text_lower = text.lower()
    for word in UNSAFE_KEYWORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            found.append(word)
    return found

# ----------------------------
# 4. Main
# ----------------------------
if __name__ == "__main__":
    video_path = input("Enter video file path: ").strip()

    print("⏳ Extracting text from video frames...")
    frame_text = extract_text_from_frames(video_path)

    print("⏳ Extracting text from video audio...")
    audio_text = extract_text_from_audio(video_path)

    # Merge OCR + ASR
    full_text = (frame_text + " " + audio_text).strip()

    # Show full extracted text
    print("\n📜 Full Extracted Text:\n", full_text if full_text else "[No text found]")

    # Find unsafe words
    unsafe_words = find_unsafe_words(full_text)
    print("\n🚨 Unsafe Words Found:", unsafe_words if unsafe_words else "[None]")

    # Decide Safe or Unsafe
    if unsafe_words:
        status = "Unsafe"
        confidence = 0.99
    else:
        status = "Safe"
        confidence = 1.0

    print("\nSafe or Unsafe:", status)
    print("Confidence:", confidence)
