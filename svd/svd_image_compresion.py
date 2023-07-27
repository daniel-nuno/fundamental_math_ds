#%%
from PIL import Image as im
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import data
from skimage.color import rgb2gray
from skimage.util.dtype import img_as_float
plt.style.use("fivethirtyeight")
#%%
gray_images = {
    "cat":rgb2gray(img_as_float(data.chelsea()))
}

color_images = {
    "cat":img_as_float(data.chelsea())
}

def compress_svd(image, k):
    U, s, V = svd(image, full_matrices=False)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    return reconst_matrix, U, s, V

def compress_show_gray_images(img_name, k):
    image = gray_images[img_name]
    original_shape = image.shape
    reconst_img, s = compress_svd(image, k)
    fig, axes = plt.subplots(1,2,figsize=(8,5))
    axes[0].plot(s)
    compression_ratio = 100 * (k * (original_shape[0] + original_shape[1]) + k) / (original_shape[0] * original_shape[1])
    axes[1].set_title('compression ratio={:.2f}'.format(compression_ratio)+'%')
    axes[1].imshow(reconst_img, cmap='gray')
    axes[1].axis('off')
    #fig.tight_layout()

def compress_show_color_images(img_name, k):
    image = color_images[img_name]
    original_shape = image.shape
    image_reshaped = image.reshape((original_shape[0],original_shape[1]*3))
    image_reconst,_ = compress_svd(image_reshaped,k)
    image_reconst = image_reconst.reshape(original_shape)
    compression_ratio =100.0* (k*(original_shape[0] + 3*original_shape[1])+k)/(original_shape[0]*original_shape[1]*original_shape[2])
    plt.title("compression ratio={:.2f}".format(compression_ratio)+"%")
    plt.imshow(image_reconst)

# %%
#compress_show_gray_images('cat', 100)
compress_show_color_images('cat', 64)
# %%
image_input = im.open("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen4Lauterbrunnen.jpg")
image = np.asarray(image_input)
k = 1
original_shape = image.shape
image_reshaped = image.reshape((original_shape[0],original_shape[1]*3))
image_reconst, U, s, V = compress_svd(image_reshaped,k)
image_reconst = image_reconst.reshape(original_shape)
compression_ratio = 100.0 * (k*(original_shape[0] + 3*original_shape[1])+k)/(original_shape[0]*original_shape[1]*3)
plt.title("compression ratio={:.2f}".format(compression_ratio)+"%")
plt.imshow(image_reconst)
# %%
image_input = im.open("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen4Lauterbrunnen.jpg")
image = np.asarray(image_input)
plt.axis('off')
plt.imshow(image)
# %%
original_shape = image.shape
#image_reshaped = image.reshape((original_shape[0],original_shape[1]*3))
image_reshaped = image.reshape((original_shape[0]*original_shape[1], 3))
U, S, VT = svd(image_reshaped, full_matrices=False)
S = np.diag(S)
r = 3
Xapprox = U[:,:r] @ S[:r,:r] @ VT[:r,:]
#image_reconst = Xapprox.reshape(original_shape)
image_reconst = Xapprox.reshape((original_shape[0], original_shape[1], 3))
image_reconst = np.array(image_reconst, np.uint8)
plt.axis('off')
plt.title('r = ' + str(r))
plt.imshow(image_reconst)
image_output = im.fromarray(image_reconst)
img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen4Lauterbrunnen") + "r" + str(r) + ".jpg"
image_output.save(img_txt)
# %%
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()
# %%
#stackedt
image_input = im.open("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen4Lauterbrunnen.jpg")
image = np.asarray(image_input)
plt.axis('off')
plt.imshow(image)
original_shape = image.shape
#%%
r = 200
image_reconst = np.zeros(image.shape)
for i in range(3):
    U, S, VT = svd(image[:,:,i], full_matrices=False)
    S = np.diag(S)
    Xapprox = U[:,:r] @ S[:r,:r] @ VT[:r,:]
    image_reconst[:,:,i] = np.array(Xapprox, np.uint8)
    image_reconst = np.array(image_reconst, np.uint8)

plt.axis('off')
plt.title('r = ' + str(r))
plt.imshow(image_reconst)

# %%
