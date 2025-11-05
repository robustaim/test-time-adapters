"""
Kalman Filter Tracker for Temporal Consistency
Inspired by SORT (Simple Online and Realtime Tracking)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou_batch(bboxes1, bboxes2):
    """
    Compute IOU between two sets of bounding boxes.

    Args:
        bboxes1: (N, 4) array in format [x1, y1, x2, y2]
        bboxes2: (M, 4) array in format [x1, y1, x2, y2]

    Returns:
        (N, M) array of IOU values
    """
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))

    # Expand dimensions for broadcasting
    bboxes1 = np.expand_dims(bboxes1, 1)  # (N, 1, 4)
    bboxes2 = np.expand_dims(bboxes2, 0)  # (1, M, 4)

    # Compute intersection
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    intersection = w * h

    # Compute union
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    union = area1 + area2 - intersection

    iou = intersection / (union + 1e-6)

    return iou


def convert_bbox_to_z(bbox):
    """
    Convert bounding box [x1, y1, x2, y2] to [x, y, s, r] format
    where x, y is center, s is scale/area, r is aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is area
    r = w / (h + 1e-6)  # aspect ratio
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_z_to_bbox(z):
    """
    Convert [x, y, s, r] format back to [x1, y1, x2, y2]
    """
    w = np.sqrt(z[2] * z[3])
    h = z[2] / (w + 1e-6)
    x1 = z[0] - w / 2.
    y1 = z[1] - h / 2.
    x2 = z[0] + w / 2.
    y2 = z[1] + h / 2.
    return np.array([x1, y1, x2, y2]).reshape((1, 4))


class KalmanBBoxTracker:
    """
    Kalman Filter based bounding box tracker.
    
    State vector: [x, y, s, r, vx, vy, vs]
    where (x, y) is center, s is scale/area, r is aspect ratio,
    and (vx, vy, vs) are velocities.
    """
    count = 0

    def __init__(self, bbox, class_id):
        """
        Initialize a tracker using initial bounding box.

        Args:
            bbox: [x1, y1, x2, y2] format
            class_id: class label
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1],  # s = s + vs
            [0, 0, 0, 1, 0, 0, 0],  # r = r (constant)
            [0, 0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1],  # vs = vs
        ])

        # Measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.  # Higher uncertainty in scale

        # Process noise
        self.kf.P[4:, 4:] *= 1000.  # High uncertainty in initial velocities
        self.kf.P *= 10.

        self.kf.Q[-1, -1] *= 0.01  # Process noise for scale velocity
        self.kf.Q[4:, 4:] *= 0.01  # Process noise for velocities

        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBBoxTracker.count
        KalmanBBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = class_id

    def update(self, bbox, class_id):
        """
        Update the tracker with a new detection.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.class_id = class_id

    def predict(self):
        """
        Predict the next state and return predicted bbox.
        """
        # Handle invalid scale
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(convert_z_to_bbox(self.kf.x))
        return self.history[-1][0]

    def get_state(self):
        """
        Return the current bounding box estimate.
        """
        return convert_z_to_bbox(self.kf.x)[0]


class TemporalTracker:
    """
    Temporal tracker that manages multiple Kalman filter trackers.
    """
    def __init__(self, max_age=3, min_hits=1, iou_threshold=0.3):
        """
        Args:
            max_age: Maximum frames to keep a track without update
            min_hits: Minimum hits to confirm a track
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections, classes):
        """
        Update trackers with new detections.

        Args:
            detections: (N, 4) array of [x1, y1, x2, y2]
            classes: (N,) array of class labels

        Returns:
            predicted_boxes: (M, 4) array of predicted boxes for next frame
            predicted_classes: (M,) array of class labels
            matched_indices: (M,) array indicating which detection matched (-1 if no match)
        """
        self.frame_count += 1

        # Get predictions from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate detections to trackers
        if len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
                detections, trks, self.iou_threshold
            )

            # Update matched trackers
            for m in matched:
                self.trackers[m[1]].update(detections[m[0]], classes[m[0]])

            # Create new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBBoxTracker(detections[i], classes[i])
                self.trackers.append(trk)

        # Remove dead tracks
        i = len(self.trackers)
        ret_boxes = []
        ret_classes = []
        ret_matched = []

        for trk in reversed(self.trackers):
            i -= 1

            # Get current state
            d = trk.get_state()

            # Check if track is confirmed and active
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret_boxes.append(d)
                ret_classes.append(trk.class_id)
                # Find which detection matched this tracker
                matched_det = -1
                if len(detections) > 0:
                    for m in matched:
                        if m[1] == i:
                            matched_det = m[0]
                            break
                ret_matched.append(matched_det)

            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret_boxes) > 0:
            return np.array(ret_boxes), np.array(ret_classes), np.array(ret_matched)
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assign detections to tracked objects using Hungarian algorithm.

        Returns:
            matched: (N, 2) array of [detection_idx, tracker_idx]
            unmatched_detections: list of detection indices
            unmatched_trackers: list of tracker indices
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

        iou_matrix = iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            # Use Hungarian algorithm for matching
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def reset(self):
        """Reset all trackers."""
        self.trackers = []
        self.frame_count = 0
        KalmanBBoxTracker.count = 0
