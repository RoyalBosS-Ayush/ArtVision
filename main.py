import numpy as np
import cv2
import matplotlib.image as matim
from glob import glob
from tqdm import tqdm


def detail(img, s=20, r=0.2):
    detaiImg = cv2.detailEnhance(img, sigma_s=s, sigma_r=r)
    return detaiImg


def adjust_gamma(image, gamma=1):
    invGamma = 1.0 / gamma
    # lookup table
    table = np.array([((i/255)**invGamma)*255 for i in np.arange(0, 256)])
    lut_img = cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
    return lut_img


def pencil_art_image(img, ksize=21, sigmaX=9, gamma=0.1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX) # ksize = 3 to 25 and sigmax = 1 to 15
    gray_blur_divide = cv2.divide(gray, gray_blur, scale=256)
    pencil_sktech = adjust_gamma(gray_blur_divide, gamma=gamma)  # 0 - 1
    return pencil_sktech


def edge_mask(img, ksize, block_size):
    # Grayscale + MedianBlur + AdaptiveThreshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_median = cv2.medianBlur(gray, ksize)
    edges = cv2.adaptiveThreshold(gray_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, block_size, ksize)
    return edges


def kmeans_cluster(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(
        data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


def cartoon_image(img, ksize=5, block_size=7, k=7, d=7, sigmacolor=200, sigmaspace=200):
    # step-1 Edge Mask
    # step-2 K-means Clustering
    # step-3 Bilateral Filter
    edgeMask = edge_mask(img, ksize, block_size)
    cluster_img = kmeans_cluster(img, k)
    bilateral = cv2.bilateralFilter(
        cluster_img, d=d, sigmaColor=sigmacolor, sigmaSpace=sigmaspace)
    cartoon = cv2.bitwise_and(bilateral, bilateral, mask=edgeMask)
    return cartoon


images = glob("./images/*.jpg")

for index, imagePath in tqdm(enumerate(images), desc="Progress", total=len(images)):
    img = matim.imread(imagePath)

    det = detail(img)
    matim.imsave(f"./output/{index+1}-det.jpg", det)

    pen = pencil_art_image(img, 7, 5, 0.23)
    cv2.imwrite(f"./output/{index+1}-pen.jpg", pen)

    cat = cartoon_image(img, 5, 7, 6, 9, 148, 149)
    matim.imsave(f"./output/{index+1}-cat.jpg", cat)
