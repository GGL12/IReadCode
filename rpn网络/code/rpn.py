import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
import numpy as np
import Helpers

def generate_base_anchors(stride, ratios, scales):
    '''
    根据滑动的窗口，及其不同的长宽比例和面积大小生成单个窗口的ratios*scale个候选框
    '''
    center = stride // 2
    base_anchors = []
    for scale in scales:
        for ratio in ratios:
            box_area = scale ** 2
            w = round((box_area / ratio) ** 0.5)
            h = round(w * ratio)
            x_min = center - w / 2
            y_min = center - h / 2
            x_max = center + w / 2
            y_max = center + h / 2
            base_anchors.append([x_min, y_min, x_max, y_max])
    return np.array(base_anchors, dtype=np.float32)

def calculate_iou(anc, gt):
    '''
        1. 去除没有交换的anchor
        2. 计算iou值
        3. 后期根据iou更新生成anchor的label值
        4. iou公式：(anchor_1 ^ anchor_2) / (anchor_1 + anchor_2)
    '''
    ### Ground truth box x1, y1, x2, y2
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1
    gt_area = gt_width * gt_height
    ### Anchor x1, y1, x2, y2
    anc_x1, anc_y1, anc_x2, anc_y2 = anc
    anc_width = anc_x2 - anc_x1
    anc_height = anc_y2 - anc_y1
    anc_area = anc_width * anc_height
    ### Possible intersection
    x_top = max(gt_x1, anc_x1)
    y_top = max(gt_y1, anc_y1)
    x_bottom = min(gt_x2, anc_x2)
    y_bottom = min(gt_y2, anc_y2)
    ### Check intersection
    if x_bottom < x_top or y_bottom < y_top:
        return 0.0
    ### Calculate intersection area
    intersection_area = (x_bottom - x_top) * (y_bottom - y_top)
    ### Calculate union area
    union_area = gt_area + anc_area - intersection_area
    # Intersection over Union
    return intersection_area / union_area

def generate_iou_map(anchors, gt_boxes, img_boundaries):
    '''
        1. 去除超过边际的anchor
        2. 对多个gt下的anchor生成对应的iou映射表
    '''
    anchor_count = anchors.shape[0]
    gt_box_count = len(gt_boxes)
    iou_map = np.zeros((anchor_count, gt_box_count), dtype=np.float32)
    for anc_index, anchor in enumerate(anchors):
        if (
            anchor[0] < img_boundaries["left"] or
            anchor[1] < img_boundaries["top"] or
            anchor[2] > img_boundaries["right"] or
            anchor[3] > img_boundaries["bottom"]
        ):
            continue
        for gt_index, gt_box_data in enumerate(gt_boxes):
            iou = calculate_iou(anchor, gt_box_data)
            iou_map[anc_index, gt_index] = iou
    return iou_map

def get_bboxes_from_deltas(anchors, deltas):
    '''
        1. 功能相似于 get_deltas_from_bboxes（）函数
        2. 处理生成的acnhor与gt相对比例，用log放缩了。
    '''
    bboxes = np.zeros(anchors.shape, dtype=np.float32)
    #
    all_anc_width = anchors[:, 2] - anchors[:, 0]
    all_anc_height = anchors[:, 3] - anchors[:, 1]
    all_anc_ctr_x = anchors[:, 0] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[:, 1] + 0.5 * all_anc_height
    #
    all_bbox_width = np.exp(deltas[:, 2]) * all_anc_width
    all_bbox_height = np.exp(deltas[:, 3]) * all_anc_height
    all_bbox_ctr_x = (deltas[:, 0] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[:, 1] * all_anc_height) + all_anc_ctr_y
    #
    bboxes[:, 0] = all_bbox_ctr_x - (0.5 * all_bbox_width)
    bboxes[:, 1] = all_bbox_ctr_y - (0.5 * all_bbox_height)
    bboxes[:, 2] = all_bbox_width + bboxes[:, 0]
    bboxes[:, 3] = all_bbox_height + bboxes[:, 1]
    #
    return bboxes

def get_deltas_from_bboxes(anchors, gt_boxes, pos_anchors):
    '''
        pos_anchors里面包含两个部分：自身anchor的index和对应类别的index
    '''
    bbox_deltas = np.zeros(anchors.shape, dtype=np.float32)
    for pos_anchor in pos_anchors:
        index, gt_box_index = pos_anchor
        anchor = anchors[index]
        gt_box_data = gt_boxes[gt_box_index]
        #
        anc_width = anchor[2] - anchor[0]
        anc_height = anchor[3] - anchor[1]
        anc_ctr_x = anchor[0] + 0.5 * anc_width
        anc_ctr_y = anchor[1] + 0.5 * anc_height
        #
        gt_width = gt_box_data[2] - gt_box_data[0]
        gt_height = gt_box_data[3] - gt_box_data[1]
        gt_ctr_x = gt_box_data[0] + 0.5 * gt_width
        gt_ctr_y = gt_box_data[1] + 0.5 * gt_height
        #
        delta_x = (gt_ctr_x - anc_ctr_x) / anc_width
        delta_y = (gt_ctr_y - anc_ctr_y) / anc_height
        delta_w = np.log(gt_width / anc_width)
        delta_h = np.log(gt_height / anc_height)
        #
        bbox_deltas[index] = [delta_x, delta_y, delta_w, delta_h]
    #
    return bbox_deltas