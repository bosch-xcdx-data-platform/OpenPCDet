import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import numpy as np


import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Run in headless mode
import matplotlib.pyplot as plt
from pcdet.utils import box_utils

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
def get_consistent_corners(boxes: np.ndarray) -> np.ndarray:
    """
    Compute 3D box corners with consistent ordering based on global orientation.
    Uses PCA to determine the true "front" of the box in global space.
    
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, yaw]
    Returns:
        corners_3d: (N, 8, 3) with consistent corner ordering:
            0: front-left-bottom
            1: front-right-bottom
            2: rear-right-bottom
            3: rear-left-bottom
            4: front-left-top
            5: front-right-top
            6: rear-right-top
            7: rear-left-top
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 8, 3))

    # Standard box template (before rotation/translation)
    template = np.array([
        [ 0.5,  0.5, -0.5],  # Will become front-left-bottom
        [ 0.5, -0.5, -0.5],  # front-right-bottom
        [-0.5, -0.5, -0.5],  # rear-right-bottom
        [-0.5,  0.5, -0.5],  # rear-left-bottom
        [ 0.5,  0.5,  0.5],  # front-left-top
        [ 0.5, -0.5,  0.5],  # front-right-top
        [-0.5, -0.5,  0.5],  # rear-right-top
        [-0.5,  0.5,  0.5]   # rear-left-top
    ])

    all_corners = []
    for box in boxes:
        x, y, z, dx, dy, dz, yaw = box
        
        # 1. Create rotation matrix from yaw
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])
        
        # 2. Scale the template by box dimensions
        scaled_template = template * np.array([dx, dy, dz])
        
        # 3. Rotate and translate the corners
        rotated_corners = np.dot(scaled_template, rotation.T)
        global_corners = rotated_corners + np.array([x, y, z])
        
        # 4. Use PCA to find the true "front" direction in global space
        #    (more robust than just using yaw when ego is moving)
        centered = global_corners - global_corners.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # The principal component is the longest dimension of the box
        principal_dir = eigvecs[:, np.argmax(eigvals)]
        
        # 5. Determine which corner is truly the front-left
        # Project all corners onto principal direction to find front
        proj = np.dot(global_corners, principal_dir)
        front_mask = proj > np.median(proj)  # Front half of points
        
        # For front corners, find the one with max left component
        front_corners = global_corners[front_mask]
        left_dir = np.array([-principal_dir[1], principal_dir[0], 0])  # Perpendicular to principal
        left_proj = np.dot(front_corners, left_dir)
        front_left_idx = np.where(front_mask)[0][np.argmax(left_proj)]
        
        # 6. Rotate template indices so front-left is first
        # The original template assumes index 0 is front-left
        # So we need to find how much to rotate the indices
        roll_amount = -front_left_idx
        corner_order = np.roll(np.arange(8), roll_amount)
        
        # 7. Apply consistent ordering
        ordered_corners = global_corners[corner_order]
        all_corners.append(ordered_corners)

    return np.stack(all_corners, axis=0)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def angle_difference(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
def match_boxes(prev_boxes, curr_boxes, prev_labels, curr_labels, dist_thresh=0.5):
    matches = [-1] * len(curr_boxes)
    if prev_boxes is None:
        return matches

    used = set()
    for i, curr in enumerate(curr_boxes):
        curr_center = curr[:3]
        curr_label = curr_labels[i]
        min_dist = float('inf')
        best_match = -1
        for j, prev in enumerate(prev_boxes):
            if j in used or curr_label != prev_labels[j]:
                continue
            prev_center = prev[:3]
            dist = np.linalg.norm(curr_center - prev_center)
            if dist < min_dist and dist < dist_thresh:
                min_dist = dist
                best_match = j
        if best_match != -1:
            matches[i] = best_match
            used.add(best_match)
    return matches

def smooth_boxes(prev_boxes, curr_boxes, prev_labels, curr_labels,
                 position_alpha=0.6, size_alpha=0.9, yaw_alpha=0.5):
    """Smooth bounding boxes over time with correct yaw handling."""
    smoothed = []
    matches = match_boxes(prev_boxes, curr_boxes, prev_labels, curr_labels)

    for i, match_idx in enumerate(matches):
        curr = curr_boxes[i]

        if prev_boxes is not None and match_idx != -1:
            prev = prev_boxes[match_idx]

            smoothed_pos = position_alpha * prev[:3] + (1 - position_alpha) * curr[:3]
            smoothed_size = size_alpha * prev[3:6] + (1 - size_alpha) * curr[3:6]

            # --- YAW SMOOTHING WITH WRAP ---
            prev_yaw = prev[6]
            curr_yaw = curr[6]
            
            # Normalize angle difference to [-pi, pi]
            delta_yaw = (curr_yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi

            # Apply smoothing to delta
            smoothed_yaw = prev_yaw + (1 - yaw_alpha) * delta_yaw
            smoothed_yaw = (smoothed_yaw + np.pi) % (2 * np.pi) - np.pi  # Re-wrap

            smoothed_box = np.hstack([smoothed_pos, smoothed_size, smoothed_yaw])
        else:
            smoothed_box = curr

        smoothed.append(smoothed_box)

    return np.array(smoothed)

def transform_to_global(boxes_local, pose):
    """Convert boxes from local to global coordinates."""
    if len(boxes_local) == 0:
        return boxes_local

    global_boxes = []
    yaw_offset = np.arctan2(pose[1, 0], pose[0, 0])
    
    for box in boxes_local:
        center_local = np.append(box[:3], 1)
        center_global = (pose @ center_local)[:3]
        global_yaw = box[6] + yaw_offset
        global_yaw = (global_yaw + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        global_boxes.append(np.hstack([center_global, box[3:6], global_yaw]))
    return np.array(global_boxes)

def transform_to_local(boxes_global, pose):
    """Convert boxes from global to local coordinates."""
    if len(boxes_global) == 0:
        return boxes_global

    inv_pose = np.linalg.inv(pose)
    yaw_offset = np.arctan2(pose[1, 0], pose[0, 0])

    local_boxes = []
    for box in boxes_global:
        center_global = np.append(box[:3], 1)
        center_local = (inv_pose @ center_global)[:3]
        local_yaw = box[6] - yaw_offset
        local_yaw = (local_yaw + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        local_boxes.append(np.hstack([center_local, box[3:6], local_yaw]))
    return np.array(local_boxes)

# from AB3DMOT.AB3DMOT_libs.model import AB3DMOT
# from easydict import EasyDict

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    print(f'Number of files in {args.data_path}: {len(glob.glob(str(Path(args.data_path) / f"*{args.ext}")))}')

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, 
        class_names=cfg.CLASS_NAMES, 
        training=False,
        root_path=Path(args.data_path), 
        ext=args.ext, 
        logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    prev_global_boxes = None
    prev_labels = None
    alpha = 0.6

    save_dir = Path(args.data_path).parent / "predictions"
    save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample index: {idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            pred_dicts, _ = model.forward(data_dict)
            pred = pred_dicts[0]

            boxes = pred['pred_boxes'].cpu().numpy()
            scores = pred['pred_scores'].cpu().numpy()
            labels = pred['pred_labels'].cpu().numpy()

            # Load ego pose
            pose_path = Path(args.data_path.replace("bin/", "")) / f"{idx:06d}_adma.npy"
            pose = np.load(pose_path) if pose_path.exists() else np.eye(4)

            # Convert to global
            global_box_coords = transform_to_global(boxes, pose)

            # Get consistent corners in global coordinates (before smoothing)
            corners_3d_global = get_consistent_corners(global_box_coords)

            # Smooth global boxes
            smoothed_global_boxes = smooth_boxes(prev_global_boxes, global_box_coords, prev_labels, labels, position_alpha=alpha)

            # Convert smoothed boxes back to local for saving
            boxes_local = transform_to_local(smoothed_global_boxes, pose)

            # Transform corners back to local for visualization
            inv_pose = np.linalg.inv(pose)
            corners_3d_local = []
            for corners in corners_3d_global:
                corners_h = np.hstack([corners, np.ones((8, 1))])
                transformed = (inv_pose @ corners_h.T).T[:, :3]
                corners_3d_local.append(transformed)
            corners_3d_local = np.array(corners_3d_local)

            # Save prediction
            np.savez(save_dir / f"pred_{idx:06d}.npz", boxes=boxes_local, scores=scores, labels=labels)

            # Visualize in local frame
            points = data_dict['points'][:, 1:4].cpu().numpy()
            save_bev(points, corners_3d_local, save_dir / f"scene_{idx:06d}.png", scores, labels)
            logger.info(f"✅ Saved predictions and visualization for index {idx} with {len(boxes)} boxes at {save_dir}/scene_{idx:06d}.png")

            # Update tracking state
            prev_global_boxes = smoothed_global_boxes
            prev_labels = labels

    logger.info('✅ Demo done.')


import matplotlib
matplotlib.use('Agg')  # Ensures headless mode
import matplotlib.pyplot as plt

def save_bev(points, corners_3d, save_path, scores, labels):
    """
    Save a BEV image with points and 3D bounding boxes and orientation indicators.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points[:, 0], points[:, 1], s=0.1, c='gray')

    if len(corners_3d) > 0:
        for corners in corners_3d:
            # Draw the box bottom face (indices 0-3)
            x = corners[[0, 1, 2, 3, 0], 0]
            y = corners[[0, 1, 2, 3, 0], 1]
            ax.plot(x, y, color='red', linewidth=2)

            # Draw a direction arrow from center to front (midpoint between corners 4 and 5)
            center = np.mean(corners, axis=0)
            front = (corners[4] + corners[5]) / 2.0  # front edge (direction)
            # ax.arrow(
            #     center[0], center[1],
            #     front[0] - center[0], front[1] - center[1],
            #     head_width=0.3, head_length=0.5,
            #     fc='blue', ec='blue'
            # )

    ax.set_xlim(0, 40)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.set_title("BEV: Points + Boxes")
    ax.grid(True)
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    main()
