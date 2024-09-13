## Overview

This project is a fun personal experiment that combines Computer Vision and interactive GUI development. By uploading an image of a Telegraph T2 crossword puzzle, the program processes the image, identifies the grid and clues, and generates an interactive interface where you can solve the crossword directly on your machine. The project utilizes the YOLO (You Only Look Once) object detection model to recognize the crossword grid and related clues. Currently, the model is specifically trained to work with Telegraph T2 crosswords.

## Features

- **Image Upload**: Upload an image of a Telegraph T2 crossword puzzle.
- **Interactive GUI**: The crossword is rendered into an interactive grid that allows you to play on your local machine.
- **Computer Vision**: Uses YOLO object detection to identify the crossword's layout and clues from the uploaded image.
- **Real-Time Feedback**: As you fill in the crossword, the interface updates interactively.
- **Customizable Interface**: The GUI layout can be adjusted based on your preferences.

## Installation

To run the project locally, you'll need to clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <project-directory>
pip install -r requirements.txt
```

## Requirements
Python 3.7+
OpenCV
PyTorch (for YOLO)
Tkinter (for GUI)
Install all dependencies using the requirements.txt file.

## Usage
Upload a valid image of a Telegraph T2 crossword puzzle.
The program will process the image, detect the crossword grid and clues, and generate an interactive GUI.
You can then play the crossword directly in the interface.
To start the application, run the following command:

```
python app.py
```

## Limitations
Model Compatibility: The YOLO model is specifically trained to detect Telegraph T2 crosswords. Other crossword formats are not supported.
Image Quality: High-quality images work best for accurate grid and clue detection. Poor lighting or low resolution may affect performance.
Puzzle Layout: Variations in the grid layout may lead to incorrect detection of certain squares or clues.

## Future Plans
Extend the model to support multiple types of crosswords from various newspapers.
Improve detection accuracy for low-quality or non-standard images.
Add features like saving progress and loading custom crossword puzzles.

## Contributing
If you'd like to contribute to this project, feel free to submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
