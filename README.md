# Hand Gesture Background Changer

A real-time computer vision application built with Python that allows users to seamlessly change their camera background using defined hand gestures for navigation and selection.

## ✨ Features

* **Gesture Control:** Navigate the background carousel using two-finger swipe gestures.
* **Instant Selection:** Select and apply a new background using a one-finger pointing gesture.
* **Live Feed Default:** Starts with your original camera feed as the background (index 0).
* **UI Carousel:** Displays a navigable list of available backgrounds with `<` and `>` indicators at the bottom of the video frame.

## ⚙️ Prerequisites

* [Python 3.8+](https://www.python.org/downloads/)
* A working webcam/camera.

## 📦 Installation and Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/HandGestureBackgroundChanger.git](https://github.com/your-username/HandGestureBackgroundChanger.git)
cd HandGestureBackgroundChanger
```
### 2. Install Dependencies
It is highly recommended to use a virtual environment (venv).

```sh
# Create and activate environment
python -m venv venv
source venv/bin/activate
# Install packages
pip install opencv-python mediapipe numpy
```
### 3. Add Custom Backgrounds

- Create a folder named Backgrounds in the root directory if it doesn't exist.
- Place your image files (.jpg, .png, .jpeg) inside the Backgrounds/ folder.

## ▶️ How to Run the Application
Execute the main Python script from the project root directory:

```sh
python main.py
```

## 🖐️ Gesture Guide
The system uses two main modes: Live Feed Mode (default) and Custom Background Mode (UI active).

| Gesture               | Action                     | Mode Transition          | Description                                                                 |
|-----------------------|----------------------------|--------------------------|-----------------------------------------------------------------------------|
| One-Finger/Two-Finger Select     | Toggle UI ON / Select BG   | Live Feed → Custom       | Selects the first custom background and activates the UI.                  |
| Two-Finger Swipe Left | Scroll Next (>)            | Custom                   | Cycles to the next background in the carousel.                              |
| Two-Finger Swipe Right| Scroll Previous (<)        | Custom                   | Cycles to the previous background in the carousel.                          |
| Fold-Finger Select     | Apply Final Background     | Custom                   | Confirms the highlighted background and sets it as the active background.  |

## Project Structure

```py
/HandGestureBackgroundChanger
|-- main.py                     # Main application loop, UI, and state logic
|-- gesture_recognizer.py       # Gesture detection (Swipe/Select)
|-- segmentation_processor.py   # Image blending and mask handling
|-- Backgrounds/                # Folder for custom background images
```

## Technical Details
This project is built on the following core components:

- MediaPipe Segmentation: Provides the high-quality mask needed to isolate the user from the background.

- MediaPipe Hands: Used to track 21 3D landmarks per hand to interpret the user's intent.

- Alpha Blending: NumPy is used to perform the blending calculation: $ \text{Output} = (\text{Foreground} \times \text{Mask}) + (\text{Background} \times (1 - \text{Mask})) $

- Stability Control: A frame-based cooldown system is implemented in main.py to prevent rapid, accidental gesture triggering.

##  Acknowledgements
- Built with [OpenCV](https://opencv.org/)
- Powered by [Google's MediaPipe Framework](https://ai.google.dev/edge/mediapipe/solutions/guide)
