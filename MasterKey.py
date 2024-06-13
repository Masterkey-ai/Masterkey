import cv2
import time
import math as m
import mediapipe as mp

def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    return int(180 / m.pi * theta)

def sendWarning():
    print("Warning: Bad posture detected for over 3 minutes!")

# Initialize posture variables
good_frames = 0
bad_frames = 0
font = cv2.FONT_HERSHEY_SIMPLEX
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

print('Processing..')
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture image")
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lm = keypoints.pose_landmarks
    if lm:
        lmPose = mp_pose.PoseLandmark
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * width)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * height)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * width)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * height)
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * width)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * height)
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * width)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * height)
        spine_x = int(lm.landmark[lmPose.NOSE].x * width)
        spine_y = int(lm.landmark[lmPose.NOSE].y * height)
        l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * width)
        l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * height)
        r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * width)
        r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * height)

        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        if offset < 100:
            cv2.putText(image, f'{int(offset)} Aligned', (width - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, f'{int(offset)} Not Aligned', (width - 150, 30), font, 0.9, red, 2)

        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        
        # Draw landmarks
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)
        cv2.circle(image, (spine_x, spine_y), 7, blue, -1)
        cv2.circle(image, (l_wrist_x, l_wrist_y), 7, blue, -1)
        cv2.circle(image, (r_wrist_x, r_wrist_y), 7, blue, -1)

        angle_text_string = f'Neck: {int(neck_inclination)}  Torso: {int(torso_inclination)}'
        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (spine_x, spine_y), light_green, 4)
            cv2.line(image, (l_wrist_x, l_wrist_y), (l_shldr_x, l_shldr_y), dark_blue, 4)
            cv2.line(image, (r_wrist_x, r_wrist_y), (r_shldr_x, r_shldr_y), dark_blue, 4)
        else:
            good_frames = 0
            bad_frames += 1
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (spine_x, spine_y), pink, 4)
            cv2.line(image, (l_wrist_x, l_wrist_y), (l_shldr_x, l_shldr_y), dark_blue, 4)
            cv2.line(image, (r_wrist_x, r_wrist_y), (r_shldr_x, r_shldr_y), dark_blue, 4)

        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames

        if good_time > 0:
            time_string_good = f'Good Posture Time: {round(good_time, 1)}s'
            cv2.putText(image, time_string_good, (10, height - 20), font, 0.9, green, 2)
        else:
            time_string_bad = f'Bad Posture Time: {round(bad_time, 1)}s'
            cv2.putText(image, time_string_bad, (10, height - 20), font, 0.9, red, 2)

        if bad_time > 180:
            sendWarning()

        # Write frames.
        video_output.write(image)

        # Show image
        cv2.namedWindow('Posture Detection', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Posture Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Posture Detection', image)

        # Press 'q' to exit the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
video_output.release()
cv2.destroyAllWindows()
