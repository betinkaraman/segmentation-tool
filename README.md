# segmentation-tool
This is a PyQt5-based application for image segmentation. It allows users to load images, create segmentation layers, and save the results.

## Features

- Load and display images
- Create multiple segmentation layers
- Draw and erase segmentations
- Adjust image properties (brightness, contrast, saturation, hue, sharpness)
- Save segmentations as separate layers and combined image

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/betinkaraman/segmentation-tool.git
   cd segmentation-tool
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, make sure you're in the root directory of the project (fundus-segmentation-tool), then use one of the following commands: (Running the command may vary depending on different Python environments.)

```
py -m src.main
```

or

```
python -m src.main
```


Once the application is running:

1. Click 'Load Image' to open a image.
2. Use the drawing tools to create segmentations.
3. Adjust image properties using the sliders.
4. Save your segmentations using the 'Save Segmentation' button.

## Troubleshooting

If you encounter issues running the script:

1. Ensure you're in the correct directory (segmentation-tool).
2. Verify that all required packages are installed by running `pip list` and comparing with the requirements.txt file.

If problems persist, please open an issue on the GitHub repository with details of the error you're encountering.

## License

[MIT License]
