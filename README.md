# Fall Detection Logic using YOLO26 Pose Estimation

This project utilizes computer vision to detect human falls in real-time. Instead of relying on simple object classification, this system uses **YOLOv8-Pose / YOLO26n-pose** to extract human skeleton keypoints. By analyzing the spatial relationship and proportions of these keypoints, the system can accurately differentiate between a person standing/sitting and a person who has fallen to the ground.

## How the Algorithm Works

The fall detection logic is built upon a heuristic approach, combining two primary parallel checks: **Bounding Box Aspect Ratio** and **Keypoint Vertical Positions (Y-Axis)**. A "Fall" (Danger) state is triggered only when *both* conditions are met simultaneously.

### 1. Bounding Box Aspect Ratio Analysis
When a person is standing or walking, the bounding box encompassing their body is typically vertical (Height > Width). Conversely, when a person falls and lies on the ground, the bounding box becomes horizontal (Width > Height).

* **Formula:** $Ratio = \frac{Height}{Width}$
* **Condition:** If $Ratio < 1.0$, the body is identified as being in a horizontal (prone/supine) position.

### 2. Keypoints Vertical Analysis (Y-Axis)
The aspect ratio alone is insufficient because it might trigger false positives (e.g., a person bending down to pick up an item while facing the camera). To confirm a fall, we analyze the positional relationship between the head and the hips. 

*Note: In computer vision coordinates (OpenCV), the origin `(0, 0)` is at the top-left of the screen, meaning the **Y-axis value increases downwards**.*

* **$Y_{nose}$:** The Y-coordinate of the Nose keypoint (Index 0), representing the head.
* **$Y_{hip\_left}$ & $Y_{hip\_right}$:** The Y-coordinates of the Left Hip (Index 11) and Right Hip (Index 12).
* **Average Hip Position:** $\bar{Y}_{hip} = \frac{Y_{hip\_left} + Y_{hip\_right}}{2}$
* **Condition:** If $Y_{nose} > \bar{Y}_{hip}$, it mathematically proves that the person's head is physically located *lower* (closer to the floor) than their center of mass (hips).

### 3. The Boolean Decision & State Holding
The final decision is a logical `AND` operation of the two conditions above.

```python
# The Core Logic
current_frame_is_fallen = (ratio < 1.0) and (nose_y > avg_hip_y)
