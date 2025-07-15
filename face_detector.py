import cv2

# Step A: Load the built-in Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step B: Start video capture from your webcam (device index 0)
cap = cv2.VideoCapture(0)

while True:
    # Step C: Read a single frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Step D: Convert frame to grayscale (Haar cascades work better with grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step E: Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Step F: Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Step G: Show the frame in a window
    cv2.imshow('Face Detector - Press q to Quit', frame)

    # Step H: Break loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step I: Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
