# python demo3.py --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml --ckpt /mount/OpenPCDet/models/voxel_rcnn_car.pth --data_path /mount/OpenPCDet/data/test1/bin
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Add this line for headless mode


def inspect_point_cloud(points, ext='.bin'):
    """
    Load a point cloud file and print its key statistics to understand its coordinate system.
    """

    
    print(f"Point cloud shape: {points.shape}")
    print(f"Data type: {points.dtype}")
    
    # Basic statistics for each dimension (assuming x, y, z, intensity)
    print("\n--- Min values (x, y, z, i):", points.min(axis=0))
    print("--- Max values (x, y, z, i):", points.max(axis=0))
    print("--- Mean values (x, y, z, i):", points.mean(axis=0))
    print("--- Std values (x, y, z, i):", points.std(axis=0))
    
    # Check the ground level by looking at the Z-axis
    z = points[:, 2]
    print(f"\n--- Z-axis (height) analysis:")
    print(f"    Min Z: {z.min():.2f} m")
    print(f"    Max Z: {z.max():.2f} m")
    print(f"    Mean Z: {z.mean():.2f} m")
    print(f"    % of points with Z < 0.5m (likely ground): {100 * np.sum(z < 0.5) / len(z):.1f}%")
    
    return points

def plot_raw_points(points, output_path="debug_plot.png"):
    """
    Create a simple 3D scatter plot to visually check the coordinate system.
    """
    fig = plt.figure(figsize=(15, 5))
    
    # BEV (X-Y)
    ax1 = fig.add_subplot(131)
    ax1.scatter(points[:, 0], points[:, 1], s=0.1, c=points[:, 2], cmap='viridis', alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top/BEV View (X-Y)')
    ax1.grid(True)
    ax1.axis('equal')
    
    # Side (X-Z)
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 2], s=0.1, c=points[:, 1], cmap='viridis', alpha=0.5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (X-Z)')
    ax2.grid(True)
    
    # Front (Y-Z)
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 1], points[:, 2], s=0.1, c=points[:, 0], cmap='viridis', alpha=0.5)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Front View (Y-Z)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"\nDebug plot saved to: {output_path}")
    plt.close()

def transform_points_to_opencdet_coords(points):
    """
    Transform point cloud to OpenPCDet coordinate system with robust handling
    """
    print("=== TRANSFORMATION FUNCTION CALLED ===")
    transformed_points = points.copy()
    
    z = points[:, 2]
    print(f"Original Z-range: [{z.min():.2f}, {z.max():.2f}] m")
    print(f"Original Z-mean: {z.mean():.2f} m")
    
    # --- Check for extreme/corrupted Z-values ---
    z_range = z.max() - z.min()
    
    if z_range > 50:  # Extreme range suggests corrupted data
        print(f"WARNING: Extreme Z-range detected ({z_range:.1f}m)")
        # Use median as ground estimate for corrupted data
        ground_z = np.median(z)
        print(f"Using median ground estimate: {ground_z:.2f}m")
    else:
        # Normal case: use 1st percentile for ground estimation
        ground_z = np.percentile(z, 1)
        print(f"Using 1st percentile ground estimate: {ground_z:.2f}m")
    
    # --- Apply ground height adjustment ---
    target_ground_z = -1.6
    z_offset = target_ground_z - ground_z
    transformed_points[:, 2] = z + z_offset
    
    print(f"Ground adjustment: {ground_z:.2f}m -> {target_ground_z:.2f}m (offset: {z_offset:.2f}m)")
    print(f"Transformed Z-range: [{transformed_points[:, 2].min():.2f}, {transformed_points[:, 2].max():.2f}] m")
    print("=== TRANSFORMATION COMPLETE ===\n")
    
    return transformed_points

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
        
        # Align to config's expected feature count
        expected = len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        cur = points.shape[1]
        if cur < expected:
            # pad missing features with zeros (e.g., timestamp)
            pad = np.zeros((points.shape[0], expected - cur), dtype=points.dtype)
            points = np.hstack([points, pad])
        elif cur > expected:
            # or truncate extra features if you somehow have more
            points = points[:, :expected]

        points[:, 3] = np.clip(points[:, 3], 0, 255) / 255.0

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        print("BEFORE TRANSFORMATION:")
        print(f"Z min: {points[:, 2].min():.2f}, max: {points[:, 2].max():.2f}, mean: {points[:, 2].mean():.2f}")

        input_dict["points"] = transform_points_to_opencdet_coords(input_dict["points"])
        
        print("AFTER TRANSFORMATION:")
        print(f'Z min: {input_dict["points"][:, 2].min():.2f}, max: {input_dict["points"][:, 2].max():.2f}, mean: {input_dict["points"][:, 2].mean():.2f}')

        inspect_point_cloud(input_dict["points"])
        plot_raw_points(input_dict["points"])
        return data_dict


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

def draw_bev_visualization(points, boxes, scores, labels, output_path, score_thresh=0.3):
    """
    Create BEV visualization and save as PNG
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter by score
    if len(boxes) > 0:
        mask = scores >= score_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

    # Split core box dims and optional velocity
    has_vel = boxes.shape[1] >= 9
    boxes7 = boxes[:, :7] if boxes.shape[1] >= 7 else boxes
    vels = boxes[:, 7:9] if has_vel else None

    # Points
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c='gray', alpha=0.6, label='Points')

    # Boxes
    for i, box in enumerate(boxes7):
        x, y, z, dx, dy, dz, yaw = box
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        half_dx, half_dy = dx / 2, dy / 2

        corners = np.array([
            [ half_dx,  half_dy],
            [ half_dx, -half_dy],
            [-half_dx, -half_dy],
            [-half_dx,  half_dy],
            [ half_dx,  half_dy]
        ])
        R = np.array([[cos_yaw, -sin_yaw],
                      [sin_yaw,  cos_yaw]])
        corners = corners @ R.T
        corners[:, 0] += x
        corners[:, 1] += y

        ax.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=2, alpha=0.8)

        # velocity arrow (optional)
        if has_vel:
            vx, vy = vels[i]
            ax.arrow(x, y, vx, vy, head_width=0.6, head_length=1.2, length_includes_head=True, alpha=0.9)

        # score label
        ax.text(x, y, f'{scores[i]:.2f}',
                fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'BEV Detection - {len(boxes7)} objects (score ≥ {score_thresh})')
    ax.grid(True)
    ax.set_aspect('equal')
    # nuScenes ranges are usually symmetric; tweak as you like
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.legend()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def draw_3d_visualization_simple(points, boxes, scores, labels, output_path, score_thresh=0.3):
    """
    Simple 3-view visualization using matplotlib (headless compatible)
    """
    fig = plt.figure(figsize=(15, 5))

    # Filter by score
    if len(boxes) > 0:
        mask = scores >= score_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

    boxes7 = boxes[:, :7] if boxes.shape[1] >= 7 else boxes

    # 1) BEV (X-Y) with rotated polygons (like the BEV helper)
    from matplotlib.patches import Polygon
    ax1 = fig.add_subplot(131)
    ax1.scatter(points[:, 0], points[:, 1], s=0.5, c='gray', alpha=0.6)

    for box in boxes7:
        x, y, _, dx, dy, _, yaw = box
        half_dx, half_dy = dx/2, dy/2
        corners = np.array([
            [ half_dx,  half_dy],
            [ half_dx, -half_dy],
            [-half_dx, -half_dy],
            [-half_dx,  half_dy]
        ])
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        R = np.array([[cos_yaw, -sin_yaw],
                      [sin_yaw,  cos_yaw]])
        pc = corners @ R.T + np.array([x, y])
        poly = Polygon(pc, fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(poly)

    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_title('BEV View')
    ax1.grid(True); ax1.set_xlim(-50, 50); ax1.set_ylim(-50, 50); ax1.set_aspect('equal')

    # 2) Side View (X-Z) – ignore yaw for simplicity
    from matplotlib.patches import Rectangle
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 2], s=0.5, c='gray', alpha=0.6)
    for x, _, z, dx, _, dz, _ in boxes7:
        rect = Rectangle((x - dx/2, z - dz/2), dx, dz, fill=False, edgecolor='blue', linewidth=2)
        ax2.add_patch(rect)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)'); ax2.set_title('Side View (X-Z)')
    ax2.grid(True); ax2.set_xlim(-50, 50); ax2.set_ylim(-3, 5); ax2.set_aspect('equal')

    # 3) Front View (Y-Z) – ignore yaw for simplicity
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 1], points[:, 2], s=0.5, c='gray', alpha=0.6)
    for _, y, z, _, dy, dz, _ in boxes7:
        rect = Rectangle((y - dy/2, z - dz/2), dy, dz, fill=False, edgecolor='green', linewidth=2)
        ax3.add_patch(rect)
    ax3.set_xlabel('Y (m)'); ax3.set_ylabel('Z (m)'); ax3.set_title('Front View (Y-Z)')
    ax3.grid(True); ax3.set_xlim(-50, 50); ax3.set_ylim(-3, 5); ax3.set_aspect('equal')

    plt.suptitle(f'3D Detection - {len(boxes7)} objects (score ≥ {score_thresh})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    # Create output directory
    output_dir = Path("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # Get predictions
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            
            # Get points (remove intensity)
            points = data_dict['points'][:, 1:].cpu().numpy()
            
            # Create and save visualizations
            bev_output_path = output_dir / f'bev_frame_{idx:06d}.png'
            draw_bev_visualization(points, pred_boxes, pred_scores, pred_labels, 
                                 bev_output_path, 0.1)
            
            three_d_output_path = output_dir / f'3d_frame_{idx:06d}.png'
            draw_3d_visualization_simple(points, pred_boxes, pred_scores, pred_labels,
                                       three_d_output_path, 0.1)

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()