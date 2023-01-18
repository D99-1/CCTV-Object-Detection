import cv2

# Load the image
#image = cv2.imread("image.png")
image = cv2.imread("4f.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Use the classifier to detect faces in the image
faces = face_cascade.detectMultiScale(gray_image)

# Draw a rectangle around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with the detected faces
cv2.imshow("Faces", image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
