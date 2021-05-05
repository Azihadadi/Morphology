import numpy as np
import cv2

img = cv2.imread("images/sample.png")
print(img.shape)
# draw 100 random circles
for i in range(0, 100):
    # randomly generate a radius size between 5 and 10
    # and then pick a random point on our canvas where the circle
    # will be drawn
    radius = np.random.randint(5, high=10)
    color = np.random.randint(0, high=256, size = (3,)).tolist()
    pt = np.random.randint(0, high=img.shape[0], size = (2,))

    # draw our random circle
    cv2.circle(img, tuple(pt), radius, (255,255,255), -1)

# Show our masterpiece
cv2.imshow("blobs", img)
cv2.imwrite("images/opening_sample.png", img)
cv2.waitKey(0)