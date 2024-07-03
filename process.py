import cv2
import torch
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

import cv2

def generate_frames(input_video_path, target_label):
    cap = cv2.VideoCapture(input_video_path)
    # total_object_count = 0  # Initialize total object count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)

        if target_label:
            # Draw bounding boxes for the specified target label
            frame_object_count = 0  # Count objects in the current frame
            for result in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = result
                label = model.names[int(cls)]
                if label == target_label:
                    frame_object_count += 1
                    # total_object_count += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (91, 218, 227), 2)
            # Display count for the current frame and total count
            cv2.putText(frame, f'{target_label} Count: {frame_object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (91, 218, 227), 2)
            # cv2.putText(frame, f'Total Count: {total_object_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Draw bounding boxes for all detected objects if no target label is specified
            for result in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = result
                label = model.names[int(cls)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (91, 218, 227), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

    cv2.destroyAllWindows()