from pdf2image import convert_from_path
import os
import cv2
import numpy as np
from skimage import io
import math


def get_document_bounds(image):
    """
    Get horizontal bounds of the health book pages on pdf page
    :param image: pdf page
    :return: left bound, right bound
    """
    rows = []
    x1, x2 = 0, 0
    size, _ = image.shape
    c = 0.035
    threshold = size * c

    for row in image:
        row_sum = int(sum(row) / 255)
        rows.append(row_sum)

    for i in range(len(rows)):
        if rows[i] > threshold:
            x1 = i
            break

    for i in range(len(rows)):
        if rows[-(i + 1)] > threshold:
            x2 = len(rows) - 1 - i
            break

    return x1, x2


def get_document_bounds(image):
    """
    Get horizontal bounds of the health book pages on pdf page
    :param image: pdf page
    :return: left bound, right bound
    """
    rows = []
    x1, x2 = 0, 0
    size, _ = image.shape
    c = 0.035
    threshold = size * c

    for row in image:
        row_sum = int(sum(row) / 255)
        rows.append(row_sum)

    for i in range(len(rows)):
        if rows[i] > threshold:
            x1 = i
            break

    for i in range(len(rows)):
        if rows[-(i + 1)] > threshold:
            x2 = len(rows) - 1 - i
            break

    return x1, x2


def get_image_pages(image, inverted=False):
    """
    Get all health book pages presented in the pdf page
    :param image: pdf page
    :param inverted: boolean that represents page orientation
    :return: health book page(s)
    """
    # Place page horizontally
    height, width = image.shape[:2]
    if height > width:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if inverted:
        image = cv2.rotate(image, cv2.ROTATE_180)

    original = image.copy()
    scale = 2
    new_height = round(height / scale)
    new_width = round(width / scale)

    image = cv2.resize(image, (new_width, new_height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 80)
    cannyt = np.transpose(canny)

    y1, y2 = tuple([scale * x for x in get_document_bounds(canny)])
    x1, x2 = tuple([scale * x for x in get_document_bounds(cannyt)])
    w1 = x2 - x1
    middle = x1 + round(w1 / 2)

    height, width, _ = original.shape
    original_area = height * height
    document_area = abs(y1 - y2) * abs(x1 - x2)

    pages = []

    if document_area > (original_area * 0.6):
        left_page = original[y1:y2, x1:middle]
        right_page = original[y1:y2, middle:x2]
        pages.append(left_page)
        pages.append(right_page)
    else:
        page = original[y1:y2, x1:x2]
        pages.append(page)

    return pages

def rotate_image(image, center, angle):
    """Rotate image.

    :param image: input image
    :param center: rotation center
    :param angle: angle of rotation
    :return: rotated image
    """
    rot_mat = cv2.getRotationMatrix2D(center.to_tuple(), angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result

def fix_axis_rotation(image, xx, yy):
    """Fix page rotation based on axis.

    :param image: input image
    :param xx: x-axis
    :param yy: y-axis
    :return: image with fixed rotation
    """
    # yy axis
    yy_dx = yy.delta_x()
    yy_dy = yy.delta_y()

    if yy_dx == 0 or yy_dy == 0:
        yy_angle = 0
    else:
        yy_angle = math.degrees(math.atan(yy_dx / yy_dy))

    # xx axis
    xx_dx = xx.delta_x()
    xx_dy = xx.delta_y()

    if xx_dy == 0 or xx_dx == 0:
        xx_angle = 0
    else:
        xx_angle = math.degrees(math.atan(xx_dy / xx_dx))

    rotation = (xx_angle + yy_angle) / 2

    if rotation != 0.0:
        image = rotate_image(image, xx.point1, rotation)
        xx.rotate(rotation)
        yy.rotate(rotation + 90)

    return image, xx, yy


def get_predefined_axis_values(page, gender):
    """Get predefined axis values.

    :param page: page number
    :param gender: child gender
    :return: values
    """
    values = {
        'GIRL': {
            4: (17, 0, 0, 24),
            5: (100, 40, 0, 24),
            6: (90, 5, 2, 20),
            7: (180, 75, 2, 20),
            8: (55, 30, 0, 36),
            9: (34, 12, 2, 20)
        },
        'BOY': {
            4: (17, 0, 0, 24),
            5: (100, 40, 0, 24),
            6: (105, 5, 2, 20),
            7: (195, 75, 2, 20),
            8: (55, 30, 0, 36),
            9: (36, 12, 2, 20)
        }
    }

    return values[gender][page]
def get_white_mask(image):
    """Get the pixels in the color range of whites.

    :param image: input image
    :return: result from the mask applied to the image
    """
    lower = np.array([225, 225, 225], np.uint8)
    upper = np.array([255, 255, 255], np.uint8)

    return cv2.inRange(image, lower, upper)

def get_content_boundaries(image):
    """Isolates the important content of the page.

    :param image: input image (page)
    :return: page content
    """
    original = image.copy()

    # Scaling down the image to optimize the edge detection
    scale = 5
    height, width = tuple([round(x / scale) for x in image.shape[:2]])
    image = cv2.resize(image, (width, height))

    # Remove white background
    white_mask = get_white_mask(image)
    image = cv2.bitwise_not(image, image, white_mask)

    # Convert to grayscale image and apply gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    canny = cv2.Canny(blur, 30, 10)

    # Apply closing and dilation to improve table/graphic borders' connectivity
    broken_line_h = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]], dtype=np.uint8)

    # Apply closing and dilation to improve table/graphic borders' connectivity
    broken_line_v = np.array([[0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0]], dtype=np.uint8)

    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, broken_line_h)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, broken_line_v)
    dilation = cv2.dilate(closing, (5, 5), iterations=5)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, broken_line_h)

    # Find contours
    contours, _hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Establish limits for the tables/graphics area. Between 40% and 60% of the page area
    height, width, _ = image.shape
    area = width * height
    lower_area = area * 0.4
    upper_area = area * 0.7

    # Find the best table/graph boundaries
    max_brightness = 0
    brightest_rectangle = 0, 0, width, height
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if lower_area < w * h < upper_area:
            mask = np.zeros(image.shape, np.uint8)
            mask[y:y + h, x:x + w] = image[y:y + h, x:x + w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness

    # Upscale the measures to the original image
    x, y, w, h = tuple([x * scale for x in brightest_rectangle])

    # Cut the content from the original image
    image = original[y:y + h, x:x + w]

    return image

def fix_page_rotation(image, gender):
    """Fix page rotation based on page lines.

    :param image: input image (page)
    :param gender: child gender
    :return: page with fixed rotation
    """
    if gender == 'BOY':
        mask = get_blue_mask(image)
    else:
        mask = get_pink_mask(image)

    height, width = mask.shape
    skell = np.zeros([height, width], dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while np.count_nonzero(mask) != 0:
        eroded = cv2.erode(mask, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(mask, temp)
        skell = cv2.bitwise_or(skell, temp)
        mask = eroded.copy()

    edges = cv2.Canny(skell, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=30)

    sum_angle = 0
    sum_len = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x1 - x2
            if dx != 0:
                dy = y1 - y2
                m = dy / dx
                angle = math.degrees(math.atan(m))
            else:
                angle = 90.00

            if abs(angle) > 45.00:
                angle -= 90.00

            if abs(angle) < 5.0:
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                sum_len += distance
                sum_angle += (angle * distance)

    if sum_len == 0:
        sum_len = 1

    rot_angle = (sum_angle / sum_len)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rot_angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def get_blue_mask(img):
    """Get the pixels in the color range of blues.

    :param image: input image
    :return: result from the mask applied to the image
    """

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([95, 40, 50], np.uint8)
    upper_blue = np.array([125, 255, 255], np.uint8)

    return cv2.inRange(hsv_img, lower_blue, upper_blue)

def blue_pixels_count(image):
    """Count of pixels in the color range of blues.

    :param image: input image
    :return: blue pixels count
    """
    blue_mask = get_blue_mask(image)

    return np.count_nonzero(blue_mask)

def get_pink_mask(img):
    """Get the pixels in the color range of pinks.

    :param image: input image
    :return: result from the mask applied to the image
    """

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([137, 20, 30], np.uint8)
    upper_pink = np.array([167, 255, 255], np.uint8)

    return cv2.inRange(hsv_img, lower_pink, upper_pink)

def get_gender(image):
    """Get the child gender based on health book color.

    :param image: input image
    :return: child gender
    """
    img = cv2.imread(image)
    blue = blue_pixels_count(img)
    pink = pink_pixels_count(img)

    if blue >= pink:
        return 'BOY'
    else:
        return 'GIRL'

def pink_pixels_count(image):
    """Count of pixels in the color range of pinks.

    :param image: input image
    :return: pink pixels count
    """
    pink_mask = get_pink_mask(image)

    return np.count_nonzero(pink_mask)

def getBoyCount(genders):
    boy =0;
    for string in genders:
        if string == "BOY":
            boy = boy + 1
    return boy

def getGirlCount(genders):
    girl=0
    for string in genders:
        if string == "GIRL":
            girl = girl + 1
    return girl


def pagesplitting(currentFolder):
    livros = os.listdir('.')
    for livro in livros:
        pages = convert_from_path(currentFolder + '/' + livro + '/' + livro + '.pdf', poppler_path='C:/Program Files (x86)/poppler-0.68.0/bin')
        x=0
        os.chdir(currentFolder + '/' + livro)
        if len(pages) == 4:
            for page in pages:
                x=x+1
                if x==1:
                    page.save('TABELA' + '.png', 'PNG')
                else:
                    page.save('GRAFICO ' + str(x-1) + '.png', 'PNG')
        if len(pages) == 5:
            for page in pages:
                x=x+1
                if x==1:
                    print("page 1 removed from: " + livro)
                else:
                    if x==2:
                        page.save('TABELA' + '.png', 'PNG')
                    else:
                        page.save('GRAFICO ' + str(x-2) + '.png', 'PNG')
        pathPDF = os.getcwd()
        os.remove(livro + '.pdf')
        os.chdir('..')


def gender(path):
    os.chdir(path)
    pages = os.listdir('.')

    i=0
    genderTotal = []

    for page in pages:
        #image = cv2.imread(page)
        gender = get_gender(page)

        genderTotal.append(gender)
        i=i+1

    xy = getBoyCount(genderTotal)
    xx = getGirlCount(genderTotal)

    if xy > xx:
        os.chdir('..')
        return 'M'
    else:
        os.chdir('..')
        return 'F'
