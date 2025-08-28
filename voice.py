import speech_recognition as sr
import pyttsx3
import time
import random
import subprocess
import webbrowser
import os

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Track active subprocess for Python scripts
active_subprocess = None

# Responses for specific phrases
RESPONSES = {
    "hello": [
        "Hi, I am Homi, your smart study buddy. I can scan your homework, create games from it. Let’s learn and play together!",
        "Hey, nice to meet you!"
    ],
    "hi": [
        "Hi, I am Homi, your smart study buddy. I can scan your homework, create games from it. Let’s learn and play together!",
        "Hey, nice to meet you!"
    ]
}

# General responses for phrases containing "can"
CAN_RESPONSES = [
    "Sure, happy to help!",
    "Always ready to assist you!",
    "Let's do this, I'm here for you!"
]

# Mapping of topics to files (placeholders for URLs)
TOPIC_FILES = {
    "addition": "addition.html",
    "subtraction": "subtraction.html",
    "multiplication": "multiply.html",
    "division": "division.html",
    "colours": "color.py",
    "shapes": "shapes.html",
    "organs": "science.py"
}

def speak(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def launch_file(filename, topic):
    """Launch a Python or HTML file."""
    global active_subprocess
    if not filename:
        print(f"No file specified for {topic}")
        speak(f"Sorry, no learning session is set up for {topic} yet.")
        return
    try:
        if filename.endswith(".py"):
            active_subprocess = subprocess.Popen(["python3", filename])
            print(f"Launched {filename} with PID {active_subprocess.pid}")
        elif filename.endswith(".html"):
            webbrowser.open(f"file://{os.path.abspath(filename)}")
            print(f"Opened {filename} in browser")
        else:
            print(f"Unsupported file type: {filename}")
            speak("Sorry, I can't open that file type.")
    except (subprocess.SubprocessError, FileNotFoundError):
        print(f"Failed to launch {filename}")
        speak(f"Sorry, I couldn't start the {topic} session.")

def close_game():
    """Close the active game (Python script or HTML page)."""
    global active_subprocess
    response = "Closing the game!"
    print(f"Responding: {response}")
    speak(response)
    
    # Close Python script subprocess if running
    if active_subprocess is not None:
        try:
            active_subprocess.terminate()
            active_subprocess.wait(timeout=5)  # Wait up to 5 seconds for clean exit
            print(f"Terminated subprocess with PID {active_subprocess.pid}")
            active_subprocess = None
        except subprocess.TimeoutExpired:
            active_subprocess.kill()  # Force kill if it doesn't terminate
            print(f"Forced termination of subprocess with PID {active_subprocess.pid}")
            active_subprocess = None
        except Exception as e:
            print(f"Error closing subprocess: {e}")
            speak("Sorry, I couldn't close the Python script.")
    
    # For HTML, prompt user to close the browser tab
    speak("Thank You")

def callback(recognizer, audio):
    try:
        # Convert audio to text
        text = recognizer.recognize_google(audio).lower()
        print(f"Transcribed: {text}")
        
        # Check for "close the game"
        if "close" in text.split() and "game" in text.split():
            close_game()
            return
        
        # Check for specific phrases (hello, hi)
        for trigger, responses in RESPONSES.items():
            if trigger in text.split():
                response = random.choice(responses)
                print(f"Responding: {response}")
                speak(response)
                return
        
        # Check for "can" in the transcribed text
        if "can" in text.split():
            response = random.choice(CAN_RESPONSES)
            print(f"Responding: {response}")
            speak(response)
            return
        
        # Check for "teach" and a topic
        if "teach" in text.split():
            for topic, filename in TOPIC_FILES.items():
                if topic in text.split():
                    response = f"Starting {topic} Interactive Game"
                    print(f"Responding: {response}")
                    speak(response)
                    launch_file(filename, topic)
                    return
        
    except sr.UnknownValueError:
        # Silently ignore unclear audio
        pass
    except sr.RequestError:
        # Silently ignore API errors
        pass

# Initialize recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Adjust for ambient noise
with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased duration for better calibration
    print("Listening in real-time... Speak now!")

# Start background listening
stop_listening = recognizer.listen_in_background(microphone, callback)

# Keep the script running
try:
    while True:
        time.sleep(0.1)  # Prevent CPU overload
except KeyboardInterrupt:
    stop_listening(wait_for_stop=False)
    print("Stopped listening.")
    speak("Goodbye!")
    # Clean up any running subprocess on exit
    if active_subprocess is not None:
        active_subprocess.terminate()