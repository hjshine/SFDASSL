import numpy as np
import torch


def apply_soft_nms(boxes, scores, classes, sigma=0.5, score_threshold=0.001, iou_threshold=0.3, method='gaussian'):
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = soft_nms(dets, sigma=sigma, score_threshold=score_threshold, iou_threshold=iou_threshold, method=method)

    final_boxes = boxes[keep]
    final_scores = scores[keep]
    final_classes = classes[keep]

    # # 결과를 묶어서 반환
    # results = []
    # for i in range(len(final_boxes)):
    #     results.append({
    #         'box': final_boxes[i],
    #         'score': final_scores[i],
    #         'class': final_classes[i]
    #     })

    # return dets[keep, :4], dets[keep, 4]
    return final_boxes, final_scores, final_classes

# def apply_nms(boxes, scores, classes, iou_threshold=0.3):
#     # 리스트 데이터를 numpy 배열로 변환
#     if isinstance(boxes, list):
#         boxes = np.array(boxes)
#     if isinstance(scores, list):
#         scores = np.array(scores)
#     if isinstance(classes, list):
#         classes = np.array(classes)

#     # dets 배열 생성
#     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
#     keep = nms(dets, iou_threshold=iou_threshold)

#     final_boxes = boxes[keep]
#     final_scores = scores[keep]
#     final_classes = classes[keep]

#     return final_boxes, final_scores, final_classes


def apply_nms2(boxlist, scorelist, classlist, iou_threshold):
    """Perform Non-Maximum Suppression (NMS) on the bounding boxes."""
    if len(boxlist) == 0:
        return [], [], []

    # GPU 텐서를 CPU로 이동하여 NumPy 배열로 변환
    boxes = np.array([box.cpu().numpy() for box in boxlist])
    scores = np.array([score.cpu().numpy() for score in scorelist]).flatten()
    classes = np.array([cls.cpu().numpy() for cls in classlist]).flatten()

    # Get the indices of boxes sorted by scores in descending order
    indices = np.argsort(scores)[::-1]

    selected_indices = []

    while len(indices) > 0:
        # Select the box with the highest score
        current = indices[0]
        selected_indices.append(current)

        if len(indices) == 1:
            break

        current_box = boxes[current]
        rest_indices = indices[1:]

        # Calculate IoU with the remaining boxes
        ious = compute_iou(current_box, boxes[rest_indices])
        # Select boxes with IoU less than the threshold
        indices = rest_indices[ious < iou_threshold]

    selected_boxes = boxes[selected_indices].tolist()
    selected_scores = scores[selected_indices].tolist()
    selected_classes = classes[selected_indices].tolist()

    return selected_boxes, selected_scores, selected_classes

def apply_nms(boxes, scores, classes, iou_threshold=0.3):

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
    elif isinstance(boxes, np.ndarray):
        boxes_np = boxes
        scores_np = scores

    dets = np.hstack((boxes_np, scores_np[:, np.newaxis])).astype(np.float32)
    #dets = np.hstack((boxes_np, scores_np)).astype(np.float32)
    keep = nms(dets, iou_threshold=iou_threshold)

    final_boxes = boxes[keep]
    final_scores = scores[keep]
    final_classes = classes[keep]

    # # 결과를 묶어서 반환
    # results = []
    # for i in range(len(final_boxes)):
    #     results.append({
    #         'box': final_boxes[i],
    #         'score': final_scores[i],
    #         'class': final_classes[i]
    #     })

    # return dets[keep, :4], dets[keep, 4]
    return final_boxes, final_scores, final_classes

def soft_nms(dets, sigma=0.5, score_threshold=0.001, iou_threshold=0.3, method='linear'):
    """
    Perform Soft-NMS on the bounding boxes.
    
    Arguments:
    dets -- np.array of shape (N, 5), where N is the number of boxes, and each box is represented by 
            [x1, y1, x2, y2, score]
    sigma -- float, standard deviation for Gaussian method
    score_threshold -- float, threshold for discarding boxes
    iou_threshold -- float, IoU threshold for suppressing boxes
    method -- str, one of 'linear', 'gaussian', or 'original'
    
    Returns:
    keep -- list of indices of the kept boxes
    """
    N = dets.shape[0]
    for i in range(N):
        max_pos = i
        max_score = dets[i, 4]
        
        # Get the box with the highest score
        for j in range(i + 1, N):
            if dets[j, 4] > max_score:
                max_pos = j
                max_score = dets[j, 4]
        
        # Swap the highest score box with the box at index i
        dets[i], dets[max_pos] = dets[max_pos].copy(), dets[i].copy()
        
        # Compute IoU of the remaining boxes with the box at index i
        pos = i + 1
        while pos < N:
            iou = compute_iou(dets[i, :4], dets[pos, :4])
            
            if method == 'linear':
                if iou > iou_threshold:
                    dets[pos, 4] *= (1 - iou)
            elif method == 'gaussian':
                dets[pos, 4] *= np.exp(- (iou ** 2) / sigma)
            else:  # 'original' NMS
                if iou > iou_threshold:
                    dets[pos, 4] = 0
            
            # Remove boxes below the score threshold
            if dets[pos, 4] < score_threshold:
                dets[pos], dets[N - 1] = dets[N - 1].copy(), dets[pos].copy()
                N -= 1
                pos -= 1
            
            pos += 1
    
    return [i for i in range(N)]


def nms(dets, iou_threshold=0.3):
    """
    Perform Non-Maximum Suppression (NMS) on the bounding boxes.
    
    Arguments:
    dets -- np.array of shape (N, 5), where N is the number of boxes, and each box is represented by 
            [x1, y1, x2, y2, score]
    iou_threshold -- float, IoU threshold for suppressing boxes
    
    Returns:
    keep -- list of indices of the kept boxes
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # Sort the boxes by score in descending order
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        area1 = (x2[i] - x1[i]) * (y2[i] - y1[i])
        area2 = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
        union = area1 + area2 - intersection
        
        iou = intersection / union
        
        # Keep boxes with IoU less than the threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def compute_iou(box1, box2):
    """
    Compute the IoU of two boxes.
    
    Arguments:
    box1 -- list or np.array of shape (4,), representing [x1, y1, x2, y2]
    box2 -- list or np.array of shape (4,), representing [x1, y1, x2, y2]
    
    Returns:
    iou -- float, intersection over union
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou



