import cv2
from dotenv import load_dotenv
# RTSP stream URL
rtsp_url = "rtsp://admin:Royal1405@89.109.241.207:1001/Streaming/Channels/101/"

# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Display the resulting frame
    cv2.imshow('RTSP Stream', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()