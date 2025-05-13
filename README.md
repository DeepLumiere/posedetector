# PoseDetector using ONNX Runtime in C#

This project demonstrates real-time human pose estimation using an ONNX model (likely YOLOv8-Pose or similar) with the Microsoft.ML.OnnxRuntime in C#. It processes an image, detects human keypoints, and visualizes these keypoints and connections on the image.

## Features

* **Human Pose Estimation:** Detects 17 common human body keypoints.
* **ONNX Model Inference:** Utilizes ONNX Runtime for efficient model execution.
* **Keypoint Visualization:** Renders detected keypoints and skeletal connections on the input image.
* **Pose Classification:** Includes basic logic to classify the detected pose as "Standing," "Sitting," "Lying Down," or "Squatting/Crouching".
* **Cross-Platform Potential:** Built with .NET, allowing for potential use on Windows, Linux, and macOS (ensure ONNX Runtime native binaries are available).

## Model

This project is designed to work with ONNX pose estimation models that output heatmaps for keypoints. The example likely uses a model like **YOLOv8n-Pose**, which outputs a tensor typically of shape `[batch_size, num_keypoints*3, height, width]` or a heatmap tensor from which keypoints can be derived. The `PostProcessResults` method in `PoseHelper.cs` is tailored to process these heatmaps.

The `Sample/Media/yolov8n-pose.onnx` file (if present in your repository) would be the model used.

## Prerequisites

* [.NET SDK](https://dotnet.microsoft.com/download) (e.g., .NET 6, .NET 7, or .NET 8+)
* (Optional, for development) An IDE like Visual Studio or VS Code.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DeepLumiere/posedetector.git](https://github.com/DeepLumiere/posedetector.git)
    cd posedetector
    ```

2.  **Restore .NET dependencies:**
    The project uses NuGet packages like `Microsoft.ML.OnnxRuntime`, `System.Drawing.Common` (or alternatives for cross-platform image handling if you've updated it), and potentially an OpenCV wrapper like `OpenCvSharp`. These will be restored automatically when you build the project.
    ```bash
    dotnet restore
    ```
    *(Or restore via your IDE)*

3.  **Ensure Model Availability:**
    Make sure the ONNX model file (e.g., `yolov8n-pose.onnx`) is accessible by the application. The current code might expect it in a specific path (e.g., relative to the executable or in a `Media` folder). Adjust the model path in `Program.cs` if necessary.

    Example `modelPath` in `Program.cs` (verify this in your actual code):
    ```csharp
    // In Program.cs or relevant configuration
    string modelPath = "path/to/your/model.onnx";
    ```

## Usage

The application is a console application that typically takes an image path as a command-line argument.

1.  **Build the project:**
    ```bash
    dotnet build --configuration Release
    ```

2.  **Run the application:**
    Navigate to the build output directory (e.g., `bin/Release/netX.X/`) and run the executable, providing the path to an image:

    ```bash
    # Example from within the project's root directory after building
    # Adjust the target framework (e.g., net7.0) and executable name as needed.
    cd AIDevGallery.Sample/bin/Release/net7.0/
    ./AIDevGallery.Sample.exe "path/to/your/image.jpg"
    # or on Windows:
    # AIDevGallery.Sample.exe "path\to\your\image.jpg"
    ```
    *(Adjust `AIDevGallery.Sample` if your project/executable name is different. The path provided in the original code snippet `AIDevGallery.Sample.Utils` suggests the project might be named `AIDevGallery.Sample`.)*

    The program will:
    * Load the specified image.
    * Perform pose detection using the ONNX model.
    * Process the results to get keypoint coordinates.
    * Classify the detected pose.
    * Render the keypoints, connections, and pose label onto the image.
    * Save the output image (e.g., as `output.jpg` in the same directory as the input image or a predefined output path - check `Program.cs` for the exact output behavior).

## Code Structure (Key Components)

* **`Program.cs` (or main entry point):**
    * Handles command-line arguments (input image path).
    * Loads the ONNX model using `InferenceSession`.
    * Preprocesses the input image for the model.
    * Runs inference.
    * Calls `PoseHelper` methods for post-processing and rendering.
    * Saves the output image.
* **`PoseHelper.cs`:**
    * `PostProcessResults()`: Converts the raw heatmap output from the ONNX model into a list of (X, Y) keypoint coordinates, scaling them to the original image dimensions.
    * `RenderPredictions()`: Draws the detected keypoints, skeletal connections, and the classified pose label onto an image.
    * `DetectPose()`: Classifies the overall pose (e.g., "Standing", "Sitting", "Lying Down") based on the relative positions of the detected keypoints. This method uses relative thresholds for better scale invariance.

## Dependencies

* **Microsoft.ML.OnnxRuntime:** For loading and running ONNX models.
* **System.Drawing.Common:** (Potentially) For `Bitmap`, `Graphics` operations. *Note: This package has cross-platform limitations on non-Windows systems. Consider alternatives like ImageSharp or SkiaSharp for better cross-platform compatibility if needed.*
* **(If used) OpenCvSharp:** Or another OpenCV wrapper for image preprocessing if the model requires specific input formats not easily achieved with `System.Drawing.Common` alone.

*(Check your `.csproj` file for the exact list of dependencies.)*

## How it Works (High-Level)

1.  **Image Loading & Preprocessing:** An input image is loaded. It might be resized and normalized to match the input requirements of the ONNX pose estimation model.
2.  **Inference:** The preprocessed image tensor is fed into the ONNX model via `OnnxRuntime`. The model outputs raw predictions, often as heatmaps indicating the likelihood of each keypoint at different locations.
3.  **Post-processing (`PostProcessResults`):**
    * The heatmaps are parsed to find the location (x, y coordinates) of the highest confidence for each keypoint.
    * These coordinates, which are relative to the model's input size (e.g., 64x48 or 640x640 heatmaps), are then scaled back to the original image's dimensions.
4.  **Pose Classification (`DetectPose`):**
    * Using the scaled keypoint coordinates, this function calculates various vertical and horizontal distances between key body parts (shoulders, hips, knees, ankles).
    * It applies a set of rules with relative thresholds (e.g., comparing leg segment lengths to torso height) to determine if the person is "Lying Down," "Standing," "Sitting," or "Squatting/Crouching."
5.  **Visualization (`RenderPredictions`):**
    * The scaled keypoints are drawn as circles on a copy of the original image.
    * Lines are drawn to connect related keypoints, forming a skeleton.
    * The classified pose label is drawn on the image.
6.  **Output:** The annotated image is saved to disk.

## To-Do / Potential Improvements

* Implement batch processing for multiple images.
* Add support for video input (frame-by-frame processing).
* Improve pose classification logic for more nuanced poses or higher accuracy.
* Optimize image preprocessing and post-processing steps.
* Add more robust error handling and logging.
* Provide clear instructions for using different ONNX pose models.
* Enhance cross-platform compatibility for image manipulation (e.g., by replacing `System.Drawing.Common` if targeting non-Windows primarily).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

Distributed under the MIT License (or specify your chosen license). See `LICENSE` file for more information.
*(If you don't have a LICENSE file, consider adding one. MIT is a common and permissive choice.)*
