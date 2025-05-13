using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing; // For Bitmap, Graphics, etc.
using System.Linq;  // Required for List initialization with [] in older .NET versions if not implicitly available

namespace AIDevGallery.Sample.Utils
{
    internal class PoseHelper
    {
        public static List<(float X, float Y)> PostProcessResults(Tensor<float> heatmaps, float originalWidth, float originalHeight, float outputWidth, float outputHeight)
        {
            List<(float X, float Y)> keypointCoordinates = new List<(float X, float Y)>(); // Explicit initialization for clarity

            // Scaling factors from heatmap to original image size
            float scale_x = originalWidth / outputWidth;
            float scale_y = originalHeight / outputHeight;

            int numKeypoints = heatmaps.Dimensions[1];
            int heatmapHeight = heatmaps.Dimensions[2]; // Note: ONNX NCHW often means [batch, channels, height, width]
            int heatmapWidth = heatmaps.Dimensions[3];  // So heatmaps.Dimensions[2] is Height, heatmaps.Dimensions[3] is Width

            for (int i = 0; i < numKeypoints; i++)
            {
                float maxVal = float.MinValue;
                int maxX = 0, maxY = 0;

                for (int y = 0; y < heatmapHeight; y++) // Iterate height first
                {
                    for (int x = 0; x < heatmapWidth; x++) // Then width
                    {
                        // Assuming NCHW format: [0, keypoint_index, y_coord, x_coord]
                        float value = heatmaps[0, i, y, x];
                        if (value > maxVal)
                        {
                            maxVal = value;
                            maxX = x; // X corresponds to heatmapWidth dimension
                            maxY = y; // Y corresponds to heatmapHeight dimension
                        }
                    }
                }

                // Scale the coordinates found in the heatmap to the original image dimensions
                // Add 0.5f to use the center of the heatmap cell for slightly better accuracy
                float scaledX = (maxX + 0.5f) * scale_x;
                float scaledY = (maxY + 0.5f) * scale_y;

                keypointCoordinates.Add((scaledX, scaledY));
            }

            return keypointCoordinates;
        }

        public static Bitmap RenderPredictions(Bitmap image, List<(float X, float Y)> keypoints, float markerRatio, Bitmap? baseImage = null)
        {
            using (Graphics g = Graphics.FromImage(image))
            {
                var sourceImageForScaling = baseImage ?? image;
                var averageOfWidthAndHeight = sourceImageForScaling.Width + sourceImageForScaling.Height;
                int markerSize = Math.Max(2, (int)(averageOfWidthAndHeight * markerRatio / 2f)); // Ensure markerSize is at least 2
                Brush keypointBrush = Brushes.Red;
                using Pen linePen = new Pen(Color.Aqua, Math.Max(1, markerSize / 2f)); // Ensure pen width is at least 1

                List<(int StartIdx, int EndIdx)> connections = new List<(int, int)>
                {
                    (5, 6),   // Left shoulder to right shoulder
                    (5, 7),   // Left shoulder to left elbow
                    (7, 9),   // Left elbow to left wrist
                    (6, 8),   // Right shoulder to right elbow
                    (8, 10),  // Right elbow to right wrist
                    (11, 12), // Left hip to right hip
                    (5, 11),  // Left shoulder to left hip
                    (6, 12),  // Right shoulder to right hip
                    (11, 13), // Left hip to left knee
                    (13, 15), // Left knee to left ankle
                    (12, 14), // Right hip to right knee
                    (14, 16), // Right knee to right ankle
                    (0, 1),   // Nose to left eye
                    (0, 2),   // Nose to right eye
                    (1, 3),   // Left eye to left ear
                    (2, 4),   // Right eye to right ear
                    // Removed (0,5) Nose to Left Shoulder and (0,6) Nose to Right Shoulder as they often clutter the torso
                    // Consider adding them back if desired for specific visualizations.
                };

                foreach (var (startIdx, endIdx) in connections)
                {
                    if (startIdx < keypoints.Count && endIdx < keypoints.Count)
                    {
                        var (startPointX, startPointY) = keypoints[startIdx];
                        var (endPointX, endPointY) = keypoints[endIdx];
                        g.DrawLine(linePen, startPointX, startPointY, endPointX, endPointY);
                    }
                }

                foreach (var (x, y) in keypoints)
                {
                    g.FillEllipse(keypointBrush, x - markerSize / 2f, y - markerSize / 2f, markerSize, markerSize);
                }

                string poseLabel = DetectPose(keypoints);
                using Font font = new Font("Arial", Math.Max(8, markerSize), FontStyle.Bold); // Ensure font size is reasonable
                Brush labelBrush = Brushes.White; // Changed to white for better contrast on various backgrounds
                // Draw a semi-transparent background for the label for better readability
                SizeF labelSize = g.MeasureString(poseLabel, font);
                g.FillRectangle(new SolidBrush(Color.FromArgb(128, 0, 0, 0)), 5, 5, labelSize.Width + 10, labelSize.Height + 10);
                g.DrawString(poseLabel, font, labelBrush, new PointF(10, 10));
            }
            return image;
        }

        public static string DetectPose(List<(float X, float Y)> keypoints)
        {
            if (keypoints.Count < 17) // COCO format has 17 keypoints
                return "Unknown (Few Keypoints)";

            // Keypoint indices (COCO 17-point format)
            const int NOSE = 0;
            const int LEFT_SHOULDER = 5, RIGHT_SHOULDER = 6;
            const int LEFT_HIP = 11, RIGHT_HIP = 12;
            const int LEFT_KNEE = 13, RIGHT_KNEE = 14;
            const int LEFT_ANKLE = 15, RIGHT_ANKLE = 16;

            var nose = keypoints[NOSE];
            var leftShoulder = keypoints[LEFT_SHOULDER];
            var rightShoulder = keypoints[RIGHT_SHOULDER];
            var leftHip = keypoints[LEFT_HIP];
            var rightHip = keypoints[RIGHT_HIP];
            var leftKnee = keypoints[LEFT_KNEE];
            var rightKnee = keypoints[RIGHT_KNEE];
            var leftAnkle = keypoints[LEFT_ANKLE];
            var rightAnkle = keypoints[RIGHT_ANKLE];

            // --- Calculate average Y coordinates (image origin Y=0 at top, increases downwards) ---
            float shoulderAvgY = (leftShoulder.Y + rightShoulder.Y) / 2f;
            float hipAvgY = (leftHip.Y + rightHip.Y) / 2f;
            float kneeAvgY = (leftKnee.Y + rightKnee.Y) / 2f;
            float ankleAvgY = (leftAnkle.Y + rightAnkle.Y) / 2f;

            // --- Calculate characteristic vertical distances (absolute values) ---
            float torsoHeight = Math.Abs(hipAvgY - shoulderAvgY);
            // Stabilize torsoHeight: if too small (e.g., bad keypoints or person very foreshortened),
            // it can make relative comparisons unstable. Use a fraction of overall detected height or a fixed minimum.
            float overallPersonHeightApproximation = Math.Abs(Math.Min(leftShoulder.Y, rightShoulder.Y) - Math.Max(leftAnkle.Y, rightAnkle.Y));
            if (torsoHeight < Math.Max(0.1f * overallPersonHeightApproximation, 15f)) // If torso height is less than 10% of overall or 15px
            {
                // If torsoHeight is unreliable, other relative measures become tricky.
                // We can set a minimum, but classification might be less accurate for these cases.
                torsoHeight = Math.Max(torsoHeight, 15f); // Use at least 15px or its actual small value if > 0
                if (torsoHeight <= 1e-5) return "Unknown (Unreliable Torso Keypoints)"; // Avoid division by zero if it's still zero
            }


            float hipToKneeVertical = Math.Abs(kneeAvgY - hipAvgY);
            float kneeToAnkleVertical = Math.Abs(ankleAvgY - kneeAvgY);
            float shoulderToAnkleVertical = Math.Abs(ankleAvgY - shoulderAvgY);

            // --- Calculate characteristic horizontal distances ---
            float shoulderWidth = Math.Abs(leftShoulder.X - rightShoulder.X);
            if (shoulderWidth < 1f) shoulderWidth = 1f; // Min width for stability

            // --- Calculate alignment differences (for lying detection) ---
            float shoulderYAlignDiff = Math.Abs(leftShoulder.Y - rightShoulder.Y);
            float hipYAlignDiff = Math.Abs(leftHip.Y - rightHip.Y);

            // --- Define Relative Thresholds (these may need tuning based on your data/model) ---
            float lyingAlignmentRatio = 0.35f; // Max Y diff of shoulders/hips relative to torsoHeight
            float lyingWidthToHeightRatio = 0.9f; // shoulderWidth should be > this * shoulderToAnkleVertical
            float lyingMaxVerticalToWidthRatio = 1.5f; // shoulderToAnkleVertical should be < this * shoulderWidth (prevents far away standing as lying)


            // For Standing/Sitting distinction (based on original logic's 60px threshold)
            // Assuming an average torso might be ~2-3 times the old 60px threshold.
            // So, 60px could be ~0.33 to 0.5 of torso height. We'll use a ratio.
            float legSegmentThresholdRatioFactor = 0.40f; // This factor times torsoHeight replaces '60'
            float legSegmentThreshold = legSegmentThresholdRatioFactor * torsoHeight;

            float uprightMinAspectRatio = 1.1f; // shoulderToAnkleVertical should be > this * shoulderWidth for upright poses

            // --- Pose Detection Logic ---

            // 1. Lying Down Detection
            bool shouldersLevel = shoulderYAlignDiff < lyingAlignmentRatio * torsoHeight;
            bool hipsLevel = hipYAlignDiff < lyingAlignmentRatio * torsoHeight;
            bool widerThanTall = shoulderWidth > lyingWidthToHeightRatio * shoulderToAnkleVertical;
            bool notExcessivelyTallComparedToWidth = shoulderToAnkleVertical < lyingMaxVerticalToWidthRatio * shoulderWidth;
            // An alternative for very compressed poses: if overall vertical span is very small compared to torso itself.
            bool veryCompressedVertically = shoulderToAnkleVertical < 0.7f * torsoHeight && torsoHeight > shoulderWidth * 0.5f; // Torso has some substance

            bool isLying = (shouldersLevel && hipsLevel && (widerThanTall || veryCompressedVertically) && notExcessivelyTallComparedToWidth);

            if (isLying) return "Lying Down";

            // --- Upright Pose Checks (Standing or Sitting) ---
            bool torsoIsUpright = shoulderAvgY < (hipAvgY - 0.15f * torsoHeight); // Hips are discernibly below shoulders

            if (!torsoIsUpright)
            {
                // Could be bending over, falling, or poor keypoints.
                return "Unknown (Torso Not Upright)";
            }

            bool overallUprightPosture = shoulderToAnkleVertical > uprightMinAspectRatio * shoulderWidth;

            // 2. Standing Detection (Inspired by original: hipToKnee > 60 && kneeToAnkle > 60)
            bool standingLegs = (hipToKneeVertical > legSegmentThreshold &&
                                 kneeToAnkleVertical > legSegmentThreshold);
            // Basic leg order for standing
            bool standingLegOrder = (hipAvgY < kneeAvgY - 0.05f * torsoHeight) && // Knees below hips
                                    (kneeAvgY < ankleAvgY - 0.05f * torsoHeight); // Ankles below knees

            bool isStanding = torsoIsUpright &&
                              standingLegs &&
                              standingLegOrder &&
                              overallUprightPosture;

            if (isStanding) return "Standing";

            // 3. Sitting Detection (Inspired by original: hipToKnee < 60 && kneeToAnkle > 60)
            bool sittingLegsCondition = (hipToKneeVertical < legSegmentThreshold &&
                                         kneeToAnkleVertical > legSegmentThreshold);

            // Additional checks for a more robust sitting pose:
            // Ankles should be below knees.
            // Hips should not be significantly higher than knees (allows for different chair heights).
            bool sittingLegOrder = (kneeAvgY < ankleAvgY - 0.1f * torsoHeight) && // Ankles clearly below knees
                                   (hipAvgY < kneeAvgY + 0.6f * torsoHeight);  // Hips not excessively above knees

            bool isSitting = torsoIsUpright &&
                             sittingLegsCondition &&
                             sittingLegOrder &&
                             overallUprightPosture && // Usually, sitting poses are also "taller" than wide in a 2D projection
                             !isLying && !isStanding; // Ensure it wasn't classified already

            if (isSitting) return "Sitting";

            // Optional: Fallback for Squatting/Crouching
            bool squatThighsBent = hipToKneeVertical < legSegmentThreshold * 1.1f; // Thighs very bent
            bool squatShinsBent = kneeToAnkleVertical < legSegmentThreshold;      // Shins also bent or short vertically
            bool squatHipsLow = hipAvgY > kneeAvgY - 0.3f * torsoHeight; // Hips near or even below knee level

            if (torsoIsUpright && overallUprightPosture && squatThighsBent && squatShinsBent && squatHipsLow && !isLying && !isStanding && !isSitting)
            {
                return "Squatting/Crouching";
            }

            return "Unknown";
        }
    }
}