import os
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re

class ActionDataset(Dataset):
    def __init__(self, id_txt_path, feature_root, keypoint_root, window_size=5, step=2, max_samples=None, max_gap=1):
        self.feature_root = feature_root
        self.keypoint_root = keypoint_root
        self.window_size = window_size
        self.step = step
        self.max_samples = max_samples
        self.max_gap = max_gap

        with open(id_txt_path, 'r') as f:
            self.sample_ids = [line.strip() for line in f.readlines() if line.strip()]

        self.data = []
        self.labels = []
        self._process_data()

    def _extract_frame_num(self, filename):
        nums = re.findall(r'\d+', filename)
        return int(nums[0]) if nums else -1

    def _process_data(self):
        sample_count = 0

        for sid in self.sample_ids:
            if self.max_samples and sample_count >= self.max_samples:
                break

            label_name = sid.split('_')[0]
            label = self._get_label(label_name)

            feat_dir = os.path.join(self.feature_root, sid)
            pose_dir = os.path.join(self.keypoint_root, sid)
            if not os.path.exists(feat_dir) or not os.path.exists(pose_dir):
                continue

            feature_files = [f for f in os.listdir(feat_dir) if f.endswith('.pkl')]
            if not feature_files:
                continue

            feat_frames, features = [], []
            for f_name in feature_files:
                frame_id = self._extract_frame_num(f_name)
                if frame_id < 0:
                    continue
                with open(os.path.join(feat_dir, f_name), 'rb') as f:
                    feat = np.array(pickle.load(f), dtype=np.float32)
                    feat = feat.squeeze() if feat.ndim > 1 else feat
                    features.append(feat)
                    feat_frames.append(frame_id)

            if len(features) == 0:
                continue
            sorted_idx = np.argsort(feat_frames)
            feat_frames = np.array(feat_frames)[sorted_idx]
            features = np.array(features)[sorted_idx]

            pose_files = [f for f in os.listdir(pose_dir) if f.endswith('.json')]
            if not pose_files:
                continue

            kp_frames, keypoints = [], []
            for p_name in pose_files:
                frame_id = self._extract_frame_num(p_name)
                if frame_id < 0:
                    continue
                json_path = os.path.join(pose_dir, p_name)
                with open(json_path, 'r') as f:
                    try:
                        kp = json.load(f)
                        if isinstance(kp, list):
                            kp = np.array(kp, dtype=np.float32)

                            if kp.ndim == 3 and kp.shape[0] == 1:
                                kp = kp[0]

                            if kp.ndim == 2 and kp.shape[1] >= 2:
                                kp = kp[:, :2].flatten()
                            else:
                                print(f"⚠️ {sid}/{p_name} 点数量异常: shape={kp.shape}")
                                kp = np.zeros((36,), dtype=np.float32)
                        else:
                            print(f"⚠️ {sid}/{p_name} 格式错误 (非list)")
                            kp = np.zeros((36,), dtype=np.float32)
                    except Exception as e:
                        print(f"❌ 读取失败: {sid}/{p_name} ({e})")
                        kp = np.zeros((36,), dtype=np.float32)

                keypoints.append(kp)
                kp_frames.append(frame_id)

            sorted_idx = np.argsort(kp_frames)
            kp_frames = np.array(kp_frames)[sorted_idx]
            keypoints = np.array(keypoints)[sorted_idx]

            common_frames = np.intersect1d(feat_frames, kp_frames)
            if len(common_frames) < self.window_size:
                continue

            feat_map = {fid: f for fid, f in zip(feat_frames, features)}
            kp_map = {fid: k for fid, k in zip(kp_frames, keypoints)}
            aligned_feats = np.stack([feat_map[i] for i in common_frames], axis=0)
            aligned_kps = np.stack([kp_map[i] for i in common_frames], axis=0)

            for i in range(0, len(common_frames) - self.window_size + 1, self.step):
                frame_slice = common_frames[i:i + self.window_size]
                if np.any(np.diff(frame_slice) > self.max_gap):
                    continue

                feat_win = aligned_feats[i:i + self.window_size]
                kp_win = aligned_kps[i:i + self.window_size]
                self.data.append((feat_win, kp_win))
                self.labels.append(label)
                sample_count += 1

                if self.max_samples and sample_count >= self.max_samples:
                    break

    def _get_label(self, label_name):
        label_name = label_name.lower().strip()
        label_map = {'call': 0, 'normal': 1, 'eye': 2}
        return label_map.get(label_name, -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, keypoints = self.data[idx]
        label = self.labels[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(keypoints, dtype=torch.float32),
            label,
        )

def get_class_distribution(dataset, phase):
    labels = np.array(dataset.labels)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"📊 {phase} 数据分布:")
    label_map = {0: "call", 1: "normal", 2: "eye"}
    for u, c in zip(unique, counts):
        print(f"   {label_map.get(u, '?')}: {c}")
    return dict(zip(unique, counts))

def get_dataloaders(base_dir,
                    batch_size=32,
                    window_size=5,
                    step=2,
                    max_samples=None,
                    feature_folder="swin_v2_pool_1024"):

    id_dir = os.path.join(base_dir, "split_id")
    feature_root = os.path.join(base_dir, feature_folder)
    keypoint_root = os.path.join(base_dir, "keypoints_18")

    datasets = {
        'train': ActionDataset(os.path.join(id_dir, "train.txt"), feature_root, keypoint_root, window_size, step, max_samples),
        'val': ActionDataset(os.path.join(id_dir, "val.txt"), feature_root, keypoint_root, window_size, step, max_samples),
        'test': ActionDataset(os.path.join(id_dir, "test.txt"), feature_root, keypoint_root, window_size, step, max_samples)
    }
    dataloaders = {
        phase: DataLoader(
            datasets[phase],
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True if phase == 'train' else False,
            prefetch_factor=2 if phase == 'train' else None,
        )
        for phase in datasets
    }

    distributions = {phase: get_class_distribution(datasets[phase], phase) for phase in datasets}
    return dataloaders, distributions

def get_test_loader_only(base_dir, batch_size=32, window_size=5, step=2,
                         max_samples=None, feature_folder="swin_v2_pool_1024"):
    id_dir = os.path.join(base_dir, "split_id")
    feature_root = os.path.join(base_dir, feature_folder)
    keypoint_root = os.path.join(base_dir, "keypoints_18")

    test_dataset = ActionDataset(
        os.path.join(id_dir, "test.txt"),
        feature_root,
        keypoint_root,
        window_size,
        step,
        max_samples
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False
    )

    return {"test": test_loader}, None
