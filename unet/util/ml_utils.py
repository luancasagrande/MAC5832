import numpy as np
import cv2


def find_bbox(image, mask, area_threshold, draw=False):
    bbox_list = []
    box_areas = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Filter Contours
    filtered_contours = []
    w, h = mask.shape
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold and area < (w * h * 0.9):
            filtered_contours.append(contour)

    for i, c in enumerate(filtered_contours):
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        box_areas.append(w * h)
        bbox_list.append(rect)

        if draw:
            cv2.rectangle(image, (x, y), (x + w, y + h), 255, 5)
    return image, bbox_list, box_areas


def calcula_iou(box_A, box_B):
    """Não mude ou renomeie esta função
        Calcula o valor do "Intersection over Union" para saber se as caixa se encontram
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x, y, w, h = box_A
    boxA = (x, y, x + w, y + h)

    x, y, w, h = box_B
    boxB = (x, y, x + w, y + h)

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    if (interArea == boxBArea or interArea == boxAArea) and boxAArea != boxBArea:
        return 100.
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def draw_overlay(image, ground_truth):
    w, h, _ = image.shape
    #   ground_truth = np.where(ground_truth == 255, 255, 0).astype('uint8')

    gt = np.zeros([w, h, 3])

    gt[:, :, 2] = ground_truth
    gt[:, :, 1] = np.ones([w, h]) * 0
    gt[:, :, 0] = np.ones([w, h]) * 0
    gt = gt.astype('uint8')

    return cv2.addWeighted(image, 0.6, gt, 0.3, 0)


def calc_fitness(test, valida, nC, removeZeroFlag = False):
    test = np.array(test).astype(np.uint8)
    valida = np.array(valida).astype(np.uint8)

    classes, counts = np.unique(valida, return_counts=True)

    if(removeZeroFlag):
        classes = classes[1:]
        counts = counts[1:]
    nC = len(classes)

    numPixels = np.sum(counts)

    sumP = 0
    confMatrix = np.zeros([nC, nC])
    for i in range(0, nC):
        for j in range(0, nC):
            pixels = np.multiply(test == classes[i], valida == classes[j])

            confMatrix[i, j] = np.sum(pixels)

            if (i == j):
                sumP += np.sum(pixels)

    ef = np.zeros(nC)
    for i in range(0, nC):
        ef[i] = (np.sum(confMatrix[:, i]) * np.sum(confMatrix[i, :])) / numPixels

    kappa = (float(sumP) - np.sum(ef)) / (numPixels - np.sum(ef))
    acc = float(sumP) / numPixels

    return kappa, acc
