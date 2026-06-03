# Fall Detection Logic using YOLO Pose Estimation

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
The final decision is determined by a logical conjunction (AND operation) of the two conditions described above. Let $F$ represent the Fall detection state, where $F = 1$ (True) indicates a detected fall and $F = 0$ (False) indicates a safe state.

$$F = (Ratio < 1.0) \land (Y_{nose} > \bar{Y}_{hip})$$

To prevent the system from flickering between $F = 0$ and $F = 1$ due to slight movements or temporary keypoint occlusion after a fall, a **State Holding Mechanism** is implemented. 
If $F = 1$ is triggered at time $t$, the system forces $F = 1$ for an interval of $[t, t + \Delta t]$ (e.g., $\Delta t = 3.0$ seconds). Even if subsequent frames mathematically evaluate to $F = 0$, the system will maintain the alarm state until the duration expires, ensuring a reliable trigger for emergency notifications.

---
*This logic serves as the foundational prototype. Future enhancements may include time-series analysis on wrist/ankle keypoints ($\sigma^2$) to differentiate between a conscious fall (movement) and an unconscious fall (no movement).*
