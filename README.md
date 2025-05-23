# AI Virtual Mouse System

![Virtual Mouse Demo](https://via.placeholder.com/800x400?text=AI+Virtual+Mouse+Demo)

> Control your computer with hand gestures - no physical mouse required!

## 📋 Overview

The AI Virtual Mouse System transforms how you interact with your computer by using computer vision and machine learning to interpret hand gestures captured through your webcam. This technology eliminates the need for a physical mouse, creating a more natural and intuitive interface, especially beneficial for presentations, accessibility needs, or when a physical mouse is impractical.

## ✨ Key Features

- **Intuitive Cursor Control**: Navigate your screen using natural index finger movements
- **Full Mouse Functionality**:
  - **Left Click**: Pinch thumb and index finger
  - **Right Click**: Pinch thumb and ring finger  
  - **Scroll**: Pinch index and middle fingers, then move up/down
- **Real-time Visual Feedback**: On-screen indicators show detected gestures and operations
- **Precision Control**: Customizable control area for improved accuracy
- **Low Latency**: Optimized for responsive, real-time interaction
- **Resource Efficient**: Minimal CPU/memory footprint

## 🔧 Technical Requirements

- Python 3.7 or higher
- Webcam (built-in or external)
- Recommended: Well-lit environment for optimal hand detection

### Dependencies
```
opencv-python>=4.5.3
mediapipe>=0.8.7
pyautogui>=0.9.53
numpy>=1.20.3
```

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-virtual-mouse.git
   cd ai-virtual-mouse
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Getting Started

1. **Launch the application**
   ```bash
   python ai_virtual_mouse.py
   ```

2. **Position your hand** within the blue rectangular area displayed on the screen

3. **Control your mouse** with these gestures:

   | Gesture | Action |
   |---------|--------|
   | Point with index finger | Move cursor |
   | Pinch thumb & index finger | Left click |
   | Pinch thumb & ring finger | Right click |
   | Pinch index & middle fingers + move | Scroll up/down |

4. **Exit the application** by pressing 'q'

## 🔍 How It Works

The system pipeline:

1. **Capture**: Webcam feed is processed frame-by-frame
2. **Detection**: MediaPipe's hand landmark detection identifies 21 key points on your hand
3. **Gesture Recognition**: Custom algorithms interpret the relationships between landmarks
4. **Action Mapping**: Recognized gestures are translated into corresponding mouse operations
5. **Smoothing**: A dynamic algorithm ensures fluid cursor movement without jitter
6. **Execution**: PyAutoGUI executes the mouse operations at the system level

## ⚙️ Configuration

Edit `config.py` to customize these parameters:

```python
# Control area dimensions (in pixels)
RECT_START_X = 100
RECT_START_Y = 100
RECT_WIDTH = 500
RECT_HEIGHT = 400

# Smoothening factor (higher = smoother but more latency)
SMOOTHENING = 7

# Gesture sensitivity
CLICK_THRESHOLD = 30  # Lower = more sensitive
SCROLL_THRESHOLD = 35
```

## 🛠️ Troubleshooting

- **Poor detection**: Ensure adequate lighting and a clear background
- **Laggy performance**: Reduce the webcam resolution in the settings
- **Cursor jumps**: Increase the smoothening factor
- **Accidental clicks**: Increase the click threshold value

## 🔮 Roadmap

- [ ] Gesture customization interface
- [ ] Multi-monitor support with edge-crossing
- [ ] Keyboard shortcut simulation
- [ ] User profiles with saved preferences
- [ ] Support for left-handed users
- [ ] Advanced gesture recognition for application-specific controls
- [ ] Accessibility optimizations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) by Google for hand tracking capabilities
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse control functionality
- [OpenCV](https://opencv.org/) for computer vision processing
