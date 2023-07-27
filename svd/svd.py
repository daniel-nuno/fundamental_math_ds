#%%
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt

image_input = im.open("image.jpg")
image = np.asarray(image_input)
plt.axis('off')
plt.imshow(image)

original_shape = image.shape
image_reshaped = image.reshape((original_shape[0], original_shape[1] * 3))
U, S, VT = np.linalg.svd(image_reshaped, full_matrices=False)
S = np.diag(S)
r = 500
Xapprox = U[:,:r] @ S[:r,:r] @ VT[:r,:]
image_reconst = Xapprox.reshape(original_shape)
image_reconst = np.array(image_reconst, np.uint8)
plt.axis('off')
plt.title('r = ' + str(r))
plt.imshow(image_reconst)

plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()

image_output = im.fromarray(image_reconst)
img_txt = str("image") + "r" + str(r) + ".jpg"
image_output.save(img_txt)

