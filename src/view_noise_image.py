import cv2
import numpy as np

random_noise = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
cv2.imshow("Random Noise", random_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()
