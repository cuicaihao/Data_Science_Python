import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


img = mpimg.imread('PP.jpg')

# plt.figure()
plt.imshow(img)
plt.show()


imgtf = tf.convert_to_tensor(img)

img_gray = tf.image.rgb_to_grayscale(imgtf);

plt.figure()

plt.imshow(img_gray)
plt.show()



