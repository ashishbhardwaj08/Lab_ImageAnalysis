import cv2
import numpy as np

# SETTINGS
input_video = "videoplayback.mp4"
output_video = "tiltshift_output.mp4"

# Blur settings
blur_strength = 31        # Must be odd
mask_blur = 101           # Must be odd

# Focus band settings
focus_center = 0.5        # 0.0 = top, 1.0 = bottom
focus_height = 0.25       # Percentage of image height kept sharp

# Saturation / contrast
saturation_scale = 1.4
contrast_scale = 1.1
brightness_shift = 10

# Playback speed effect
speed_up_factor = 2       # 2 means save every 2nd frame

# OPEN VIDEO
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Reduce FPS if speeding up
output_fps = fps * speed_up_factor

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, output_fps, (width, height))

# CREATE MASK
mask = np.zeros((height, width), dtype=np.float32)

center_y = int(height * focus_center)
focus_band = int(height * focus_height)

for y in range(height):
    distance = abs(y - center_y)

    if distance <= focus_band // 2:
        value = 1.0
    else:
        fade_distance = distance - (focus_band // 2)
        max_fade = height // 2

        value = max(0.0, 1.0 - (fade_distance / max_fade))

    mask[y, :] = value

# Smooth mask edges
mask = cv2.GaussianBlur(mask, (mask_blur, mask_blur), 0)

# Convert to 3 channels
mask_3ch = cv2.merge([mask, mask, mask])

# PROCESS VIDEO
frame_index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Skip frames for miniature time-lapse effect
    if frame_index % speed_up_factor != 0:
        frame_index += 1
        continue

    # Create blurred frame
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)

    # Convert to float
    frame_float = frame.astype(np.float32) / 255.0
    blurred_float = blurred.astype(np.float32) / 255.0

    # Blend sharp + blurred image
    result = frame_float * mask_3ch + blurred_float * (1.0 - mask_3ch)

    # Convert back to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)


    # Increase saturation

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 1] *= saturation_scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


    # Increase contrast/brightness

    result = cv2.convertScaleAbs(
        result,
        alpha=contrast_scale,
        beta=brightness_shift
    )

    # Write output frame
    out.write(result)

    frame_index += 1

    print(f"Processed frame {frame_index}")

# CLEANUP
cap.release()
out.release()

print("Done!")
print("Saved to:", output_video)