import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import speech_recognition as sr

# Setup variables
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

actions_to_map = ["play", "pause", "next", "previous", "volume_up", "volume_down", "mute"]
gesture_mappings = {}
is_playing = False  

def finger_is_up(lm_tip, lm_pip):
    return lm_tip.y < lm_pip.y

def get_finger_vector(lm):
    finger_ids = {
        "thumb": (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        "index": (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        "middle": (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        "ring": (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        "pinky": (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    }
    vector = []
    for name in ["thumb", "index", "middle", "ring", "pinky"]:
        tip, pip = finger_ids[name]
        vector.append(int(finger_is_up(lm[tip], lm[pip])))
    return tuple(vector)

def execute_action(action):
    global is_playing
    if action == "play":
        if not is_playing:
            pyautogui.press("playpause")
            print(f"Action executed: {action}")
            is_playing = True
        else:
            print("Already playing. No action.")
    elif action == "pause":
        if is_playing:
            pyautogui.press("playpause")
            print(f"Action executed: {action}")
            is_playing = False
        else:
            print("Already paused. No action.")
    elif action == "next":
        pyautogui.press("nexttrack")
        print("Action executed: next track")
    elif action == "previous":
        pyautogui.press("prevtrack")
        print("Action executed: previous track")
    elif action == "volume_up":
        pyautogui.press("volumeup")
        print("Action executed: volume up")
    elif action == "volume_down":
        pyautogui.press("volumedown")
        print("Action executed: volume down")
    elif action == "mute":
        pyautogui.press("volumemute")
        print("Action executed: mute/unmute")

# Voice control thread function
def voice_control():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    global is_playing

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Voice control ready, say a command...")

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=3)
            command = recognizer.recognize_google(audio).lower()
            print(f"Voice command received: {command}")

            # Maping voice commands to actions
            if "play" in command or "start" in command:
                if not is_playing:
                    execute_action("play")
                else:
                    print("Already playing. No action.")
            elif "pause" in command or "stop" in command:
                if is_playing:
                    execute_action("pause")
                else:
                    print("Already paused. No action.")
                execute_action("pause")
            elif "next" in command or "skip" in command:
                execute_action("next")
            elif "previous" in command or "back" in command:
                execute_action("previous")
            elif "volume up" in command or "louder" in command:
                execute_action("volume_up")
            elif "volume down" in command or "quieter" in command:
                execute_action("volume_down")
            elif "mute" in command:
                execute_action("mute")
        except sr.UnknownValueError:
            print("Could not understand the command, please try again.")
            pass
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service; {e}")
            break

# Start voice control in a separate thread
voice_thread = threading.Thread(target=voice_control, daemon=True)
voice_thread.start()

# gesture setup function to map gestures to actions
def setup_gestures(max_wait=20, stable_time=1):
    print("=== Setup: Show gestures for the following actions ===")
    for action in actions_to_map:
        print(f"Show gesture for: {action.upper()}")
        captured_vector = None
        start_stable = None
        start_time = time.time()

        while True:
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark
                vector = get_finger_vector(lm)

                cv2.putText(frame, f"Detected Vector: {vector}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Show gesture for: {action.upper()}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if captured_vector != vector:
                    # New gesture detected — reset timer
                    captured_vector = vector
                    start_stable = time.time()

                elif time.time() - start_stable > stable_time:
                    gesture_mappings[captured_vector] = action
                    print(f"Gesture for '{action}' registered: {captured_vector}")
                    time.sleep(1)  # pause before next action
                    break

            else:
                cv2.putText(frame, f"Show gesture for: {action.upper()}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                captured_vector = None
                start_stable = None

            if time.time() - start_time > max_wait:
                print(f"Timeout! No stable gesture detected for '{action}' in {max_wait} seconds. Skipping.")
                break

            cv2.imshow("Gesture Setup", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Setup aborted by user.")
                exit(0)

    print("Setup complete! Starting gesture control...")
    cv2.destroyWindow("Gesture Setup")


# detect gesture and perform action if mapped
def gesture_control_loop():
    last_action_time = 0
    gesture_delay = 1.5

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark
            vector = get_finger_vector(lm)

            current_time = time.time()
            if vector in gesture_mappings and (current_time - last_action_time) > gesture_delay:
                execute_action(gesture_mappings[vector])
                last_action_time = current_time
                cv2.putText(frame, f"Action: {gesture_mappings[vector]}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(frame, f"Gesture: {vector}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Show hand gesture", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Spotify Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


setup_gestures()

gesture_control_loop()
