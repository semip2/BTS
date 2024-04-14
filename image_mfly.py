"""
Image processing functions.
M-Fly 2023-2024
"""
import math
import numpy as np
import csv
import cv2
import time
from sklearn.cluster import KMeans, SpectralClustering
from skimage.transform import resize
from skimage.util import random_noise
from scipy.spatial import distance
import skimage.filters as filters
from skimage.filters import threshold_multiotsu
from pathlib import Path
from PIL import Image
from scipy.ndimage import rotate, zoom, shift
from imageio.v3 import imread, imwrite
from colorthief_mod import ColorThief
# import rembg

def Grayscale():
    """Return function to grayscale image."""

    def _grayscale(img):
        """Return 3-channel grayscale of image."""
        # avg = np.mean(img, axis=2)
        # avg = np.round(avg)
        # avg = np.repeat(avg[...,None], 3, 2)
        # avg = avg.astype(np.uint8)

        # return avg
        
        gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        # gray = np.dot(img[...,:3], [0.5, 0.5, 0]).astype(np.uint8)
        # gray = np.dot(img[...,:3], [0, 0.5, 0.5]).astype(np.uint8)
        # gray = np.dot(img[...,:3], [0.5, 0, 0.5]).astype(np.uint8)
        # gray = np.dot(img[...,:3], [0, 0, 1]).astype(np.uint8)
        # return np.dstack([gray, gray, gray])
        return gray

    return _grayscale

def GrayscaleChannel(channel=0):
    """Return function to grayscale image."""

    def _grayscalechannel(img):
        """Return 1-channel grayscale of image based on the channel parameter."""

        arr = [0, 0, 0]
        arr[channel] = 1
        gray = np.dot(img[...,:3], arr).astype(np.uint8)
        return gray

    return _grayscalechannel


def RandomNoise(var=0.001):
    """Return function to add noise to image."""

    def _randomnoise(img):
        """Add gaussian noise to image."""
        noisy = random_noise(img, mode="gaussian", var=var)
        noisy = (noisy * 255).astype(np.uint8)
        return noisy

    return _randomnoise


def Rotate(deg=20):
    """Return function to rotate image."""

    def _rotate(img):
        """Rotate image by deg."""
        rotated = rotate(img, angle=deg, reshape=False, order=1)
        return rotated

    return _rotate


def RandomRotate(deg=20):
    """Return function to randomly rotate image."""

    def _randomrotate(img):
        """Rotate a random integer amount in the range (-deg, deg) (inclusive).

        Keep the dimensions the same and fill any missing pixels with white.
        """
        angle = np.random.randint(-deg, deg)
        return Rotate(angle)(img)

    return _randomrotate


def Zoom(zoom_factor=0.5):
    """Return function to zoom image."""

    def _zoom(img):
        """"""
        h, w = img.shape[:2]
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

        if zoom_factor < 1:

            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, order=1)

        elif zoom_factor > 1:

            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, order=1)
            # print(out.shape)
            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            if out.shape[0] != h or out.shape[1] != w:
                out = Resize()(out)
            # trim_top = ((out.shape[0] - h) // 2)
            # trim_left = ((out.shape[1] - w) // 2)
            # out = out[trim_top:trim_top+h, trim_left:trim_left+w]
            # print(out.shape)

        else:
            out = img
        
        return out

    return _zoom


def Gaussian(percent=0.25):
    """Return function to add gaussian noise to image."""

    def _gaussian(img):
        row, col, _ = img.shape
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(img.astype(np.float32), 1 - percent, percent * gaussian, percent, 0)
        return gaussian_img.astype(np.uint8)

    return _gaussian


def Invert(percent=0.25):
    """Return function to invert image."""

    def _invert(img):
        inverted = np.invert(img)
        return inverted

    return _invert


def Contrast(limit=2.0):
    """Return function to contrast image."""

    def _contrast(img):
        """
        Adjusts contrast and brightness of an uint8 image.
        contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
        brightness: [-255, 255] with 0 leaving the brightness as is
        """
        # converting to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Stacking the original image with the enhanced image
        return enhanced_img

    return _contrast


<<<<<<< HEAD
# def Mask(thresh=50, zoom=4):
#     """Return function to mask image."""

#     def _mask(img):
#         # Get zoomed img (to extract shape/char color)

#         h, w = img.shape[ : 2]
#         zoom_tuple = (zoom,) * 2 + (1,) * (img.ndim - 2)
#         zh = int(np.round(h / zoom))
#         zw = int(np.round(w / zoom))
#         top = (h - zh) // 2
#         left = (w - zw) // 2
#         img_zoomed = img[top : top + zh, left : left + zw]

#         ct = ColorThief(img_zoomed)
#         palette = ct.get_palette(color_count=2)
#         c2 = np.array(palette[1])
#         lower = c2 - thresh
#         upper = c2 + thresh
#         print(img.shape)
#         mask = cv2.inRange(img, lower, upper)
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#         return mask

#     return _mask

def Mask():
=======
def Mask(thresh=80, zoom=4):
>>>>>>> 7e3f8f0 (mfloc compatibility)
    """Return function to mask image."""

    def _mask(img):
        """Get mask of image via k++ means clustering."""
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hues = img[...,2]
        hue_pts = []
        for i in range(hues.shape[0]):
            for j in range(hues.shape[1]):
                hue_pts.append([i, j, hues[i][j]])

        hue_pts = np.array(hue_pts)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit((hue_pts[...,2]).reshape(-1, 1))
        # kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(hue_pts ** 3)
        # spectral = SpectralClustering(n_clusters=2, random_state=0, assign_labels="discretize", verbose=True).fit(hue_pts[...,2].reshape(-1, 1))

        mask = np.zeros(hues.shape)
        for i, pt in enumerate(hue_pts):
            if kmeans.labels_[i] == 1:
                mask[pt[0]][pt[1]] = 1

        return np.dstack([mask, mask, mask])

    return _mask 


def Blur(ksize=3):
    """Return function to blur image."""

    def _blur(img):
        blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return blur

    return _blur


def Resize(w=128, h=128):
    """Return function to resize image."""

    def _resize(img):
        """Resize image to w x h. (#rows, #columns)"""
        res = resize(img, (h, w), anti_aliasing=False, order=0)
        return res.astype(np.uint8)

    return _resize


def RandomResize(lower=1, upper=2):
    """Return function to randomly resize image."""

    def _randomresize(img):
        """Resize image to w x h."""
        rand_factor = np.random.random() * (upper - lower) + lower
        w = int(rand_factor * img.shape[0])
        h = int(rand_factor * img.shape[1])
        res = resize(img, (w, h), anti_aliasing=False, order=0)
        return res.astype(np.uint8)

    return _randomresize


def RandomShift(val=16):
    """Return function to randomly shift image."""

    def _randomshift(img):
        """Shift image."""
        shift_x = np.random.randint(-val, val)
        shift_y = np.random.randint(-val, val)
        res = shift(img, [shift_x, shift_y, 0], order=0)
        return res.astype(np.uint8)

    return _randomshift


def RandomZoom(lower=0.5, upper=1):
    """Return function to randomly zoom image."""

    def _randomzoom(img):
        """Zoom image in the range of [lower, upper]."""
        rand_factor = np.random.random() * (upper - lower) + lower
        res = Zoom(rand_factor)(img)
        return res.astype(np.uint8)

    return _randomzoom


def LayBackground(bg_img):
    """Return function to lay shape on background."""

    def _laybackground(img):
        """Resize image to w x h."""
        pil_image = Image.fromarray(np.uint8(img)).convert('RGBA')
        res = Image.alpha_composite(bg_img, pil_image)
        # return res.astype(np.uint8)
        return np.array(res)

    return _laybackground


def CropChar(dim=64):
    """Return function to crop img to character, returns as nxn threshold image."""

    def _cropchar(img):
        """Detects character in image and crops to nxn."""
        # grayscale/resize
        kernel = np.ones((3, 3), np.uint8)
        # img = augment(img, [LayBackground(bg_img), Resize(256, 256)])
        img = augment(img, [Resize(256, 256)])
        # img = cv2.GaussianBlur(img, (13, 13), 0)
        img_base = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        # print(img.shape)
        L = img[:, :, 0]
        A = img[:, :, 1]
        B = img[:, :, 2]

        # otsu threshold
        try:
            multiotsuL = threshold_multiotsu(L, classes=3)
        except:
            try:
                multiotsuL = threshold_multiotsu(L, classes=2)
            except:
                multiotsuL = threshold_multiotsu(L, classes=1)
        
        try:
            multiotsuA = threshold_multiotsu(A, classes=3)
        except:
            try:
                multiotsuA = threshold_multiotsu(A, classes=2)
            except:
                multiotsuA = threshold_multiotsu(A, classes=1)

        try:
            multiotsuB = threshold_multiotsu(B, classes=3)
        except:
            try:
                multiotsuB = threshold_multiotsu(B, classes=2)
            except:
                multiotsuB = threshold_multiotsu(B, classes=1)

        multiotsuL = np.digitize(L, bins=multiotsuL).astype('uint8')
        multiotsuA = np.digitize(A, bins=multiotsuA).astype('uint8')
        multiotsuB = np.digitize(B, bins=multiotsuB).astype('uint8')

        white = np.full((256, 256), 255, dtype=np.uint8)
        thresh = (multiotsuL == 0) * white
        thresh2 = (multiotsuL == 1) * white
        thresh3 = (multiotsuL == 2) * white
        thresh4 = (multiotsuA == 0) * white
        thresh5 = (multiotsuA == 1) * white
        thresh6 = (multiotsuA == 2) * white
        thresh7 = (multiotsuB == 0) * white
        thresh8 = (multiotsuB == 1) * white
        thresh9 = (multiotsuB == 2) * white

        # find contours
        image_center = np.asarray(L.shape) / 2
        image_center = tuple(image_center.astype('int32'))
        contours, _  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _  = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours3, _  = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours4, _  = cv2.findContours(thresh4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours5, _  = cv2.findContours(thresh5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours6, _  = cv2.findContours(thresh6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours7, _  = cv2.findContours(thresh7, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours8, _  = cv2.findContours(thresh8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours9, _  = cv2.findContours(thresh9, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 300
        all_contours = [contours, contours2, contours3, contours4, contours5, contours6, contours7, contours8, contours9]
        
        # compare average distances of all points in the contour to the center, choose the contour with the smallest average distance - vijay

        # get correct contour of the text
        best_i = 0
        best_n = 0
        best_distance = 5000
        best_area = cv2.contourArea(all_contours[best_n][best_i])
        draw_contours = []
        # print(len(contours))
        for n, contours in enumerate(all_contours):
            for i, contour in enumerate(contours):
                # find center of each contour
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    # print("continued")
                    continue
                center_X = int(M["m10"] / M["m00"])
                center_Y = int(M["m01"] / M["m00"])
                contour_center = (center_X, center_Y)
            
                # calculate distance to image_center
                curr_distance = (distance.euclidean(image_center, contour_center))
                if curr_distance > 100:
                    # print("too far")
                    continue
                curr_area = cv2.contourArea(contour)
                if curr_area < min_contour_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)

                # break ties between two close distances by choosing the smaller contour
                # print(curr_distance, curr_area)
                if abs(curr_distance - best_distance) < 10:
                    if curr_area < best_area and curr_area < 10000 and h > 30:
                        best_i = i
                        best_n = n
                        best_distance = curr_distance
                        best_area = curr_area
                elif curr_distance < best_distance and curr_area < 10000 and h > 30:
                    best_i = i
                    best_n = n
                    best_distance = curr_distance
                    best_area = curr_area

        # crop image to bounding rect
        x, y, w, h = cv2.boundingRect(all_contours[best_n][best_i])
        
        # add some extra padding so future padding works
        if w < 250 and h < 250:
            w += 10
            h += 10
            x = max(0, x - 5)
            y = max(0, y - 5)

        roi_image = L
        if best_n // 3 == 1:
            roi_image = A
        elif best_n // 3 == 2:
            roi_image = B
        
        roi = roi_image[y: y + h, x: x + w]

        # resize to square
        target_dim = 64
        # max out on the longer side
        if h > w:
            smaller_side = np.floor(w / h * target_dim)
            roi = augment(roi, [Resize(smaller_side, target_dim)])
        else:
            smaller_side = np.floor(h / w * target_dim)
            roi = augment(roi, [Resize(target_dim, smaller_side)])
        
        h, w = roi.shape

        roi = cv2.medianBlur(roi, 5)
        _, roi = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # add border padding (top, bottom, left, right)
        first_pad = int((target_dim - smaller_side) / 2)
        second_pad = int(np.ceil((target_dim - smaller_side) / 2))
        border_color = np.median(np.concatenate([roi[0], roi[-1], roi[:, 0], roi[:, -1]]))
        if h > w:
            roi = cv2.copyMakeBorder(roi, 0, 0, first_pad, second_pad, cv2.BORDER_CONSTANT, value=border_color)
        else:
            roi = cv2.copyMakeBorder(roi, first_pad, second_pad, 0, 0, cv2.BORDER_CONSTANT, value=border_color)
        
        # denoise
        roi = cv2.erode(roi, kernel, iterations=2)
        roi = cv2.dilate(roi, kernel, iterations=2)

        if roi.shape != (n, n):
            roi = augment(roi, [Resize(dim, dim)])

        return roi

    return _cropchar


def augment(img, transforms):
    """Augment image at filename.
        - img: numpy array of image
        - transforms: list of image transformations
            e.g. [Grayscale(), Rotate()]
    """
    out = np.copy(img)
    for transform in transforms:
        out = transform(out)
    return out


def augment_images(transforms_list, input_dir="images/",
                   output_dir="augmented/", base_aug=[],
                   read_csv="characters.csv", write_csv="augmented_characters.csv",
                   standardize=False):
    """Augment list of images and output to output_dir.
        - transforms_list: combination of image transformations
            e.g. [ [Grayscale()], [Grayscale(), Rotate()] ]
            generates 2 augmented images: one grayscaled, one grayscaled and rotated
        - input_dir: input image directory
        - output_dir: output directory for augmented images
        - base_aug: list of augmentations that apply to all transforms (e.g. Mask())
        - read_csv: csv file for input images
        - write_csv: csv file for augmented images
        - standardize: keeps original filename if True
    """
    reader = csv.DictReader(open(read_csv, "r"), delimiter=",")
    num_rows = sum(1 for row in csv.DictReader(open(read_csv, "r"), delimiter=","))
    print_point = num_rows // 20

    if not standardize:
        writer = csv.DictWriter(
            open(write_csv, "w"),
            fieldnames=["filename", "label", "numeric_label"]
        )
        writer.writeheader()

    for i, row in enumerate(reader):
        # Get csv columns
        filename = row['filename']
        label = row['label']
        numeric_label = row['numeric_label']
        path = f"{input_dir}{filename}"

        augmented_imgs = []
        image = imread(path, pilmode='RGBA')
        if len(base_aug) != 0:
            image = augment(image, base_aug)
            augmented_imgs.append(image)

        # Generate all augmentations for each image
        augmented_imgs += [augment(image, transforms) for transforms in transforms_list]

        # Write augmented images to output_dir
        if not standardize:
            for j, img in enumerate(augmented_imgs):
                fname = f"{Path(filename).stem}_aug_{j}.jpg"
                imwrite(f"{output_dir}{fname}", img, pilmode='RGB')
                writer.writerow(
                    {
                        "filename": fname,
                        "label": label,
                        "numeric_label": numeric_label
                    }
                )     
        else:
            for img in augmented_imgs:
                fname = f"{Path(filename).stem}.png"
                # print(f"{output_dir}{fname}")
                imwrite(f"{output_dir}{fname}", img)

        # Fancy progress bar :)
        if i % print_point == 0:
            # print("[%-20s] %d%%\r" % ('=' * (20 * i // num_rows), 5 * (20 * i // num_rows)), end="")
            print("[%-20s] %d%%\r" % ('=' * (i // print_point), 5 * i // print_point), end="")
    print("[%-20s] %d%%\r" % ('=' * 20, 100), end="")
    print()
