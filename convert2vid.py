import cv2
import os

video_name = 'output.avi'

images = [img for img in os.listdir(os.getcwd()) if img.endswith(".png")]
frame = cv2.imread(images[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(image))


cv2.destroyAllWindows()
video.release()
