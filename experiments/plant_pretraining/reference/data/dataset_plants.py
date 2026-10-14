import os
from pathlib import Path
import numpy as np
import random
from medpy.io import load
import torch
import pyvista
from torch.utils.data import Dataset
from torch.nn.functional import conv2d
from torchvision.transforms import Grayscale
import torch.nn.functional as F
from utils.utils import rotate_image, rotate_coordinates
from monai.transforms import GaussianSmooth, RandGaussianNoise, ScaleIntensity, Resize, SpatialPad
from PIL import Image

train_transform = []
val_transform = []


class Plants2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, transform, gaussian_augment=False, rotate=False, continuous=True, size=64, padding=5, real_set_augment=False, growth_range=[1, 3], domain_classification=-1):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform = transform
        self.gaussian_augment = gaussian_augment
        self.rotate = rotate
        self.continuous = continuous

        weights = torch.tensor([[1., 1., 1.],
                                [1., 1., 1.],
                                [1., 1., 1.]])
        self.growth_tensor = weights.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.growth_lower_bound, self.growth_upper_bound = growth_range

        self.resize = Resize(spatial_size=(size-2*padding, size-2*padding))
        self.size = size
        self.padding = padding
        self.real_set_augment = real_set_augment

        self.pad_transform = SpatialPad([size,size,size], value=-0.5)
        self.pad_transform2d = SpatialPad([size, size], value=-0.5)

        self.transform_gray = Grayscale()

        if real_set_augment:
            self.gaussian_noise = RandGaussianNoise(prob=1, std=0.2, mean=0.55)
            self.gaussian_smooth = GaussianSmooth(sigma=1)
        else:
            self.gaussian_noise = RandGaussianNoise(prob=1, std=0.4, mean=0.)
            self.gaussian_smooth = GaussianSmooth(sigma=1.2)

        self.domain_classification = domain_classification

        self.scaling = ScaleIntensity(minv=-0.5, maxv=0.5)

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)
    
    def _project_image_3d(self, sliced_data, z_pos):
        """
        Project a 2D slice into a 3D space
        Paramaters:
            sliced_data: 2D image slice. Must be a torch tensor with values in the range [-0.5, 0.5]
        Returns:
            Projected 3D image
        """

        projected_data = torch.zeros(size=(
            sliced_data.shape[0],
            sliced_data.shape[1],
            sliced_data.shape[2],
            sliced_data.shape[-1]
        ))
        cutoff_slice_data = sliced_data.clone()
        projected_data[:, :, :, round(z_pos * sliced_data.shape[1]) + 1] += cutoff_slice_data
        projected_data[:, :, :, round(z_pos * sliced_data.shape[1]) + 2] += cutoff_slice_data
        projected_data[:, :, :, round(z_pos * sliced_data.shape[1]) - 1] += cutoff_slice_data
        projected_data[:, :, :, round(z_pos * sliced_data.shape[1]) - 2] += cutoff_slice_data
        projected_data[:, :, :, round(z_pos * (sliced_data.shape[1]))] += cutoff_slice_data
        projected_data -= 0.5

        return projected_data

    def _normalize_loaded_2d(self, array_2d_or_3d):
        """
        Normalize loaded image/seg arrays to a single-channel (H, W) layout.
        Applies the same axis handling for both images and segmentations.
        """
        arr = np.asarray(array_2d_or_3d)

        if arr.ndim == 2:
            return arr

        if arr.ndim == 3:
            # Channel-first (C,H,W) -> channel-last (H,W,C)
            if arr.shape[0] <= 4 and arr.shape[-1] > 4:
                arr = np.moveaxis(arr, 0, -1)

            # Keep single channel if present
            if arr.shape[-1] == 1:
                return arr[..., 0]

            # Convert RGB/RGBA-like arrays to grayscale
            return np.array(Image.fromarray(arr).convert("L"))

        raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")

    # def __getitem__(self, idx):
    #     """[summary]

    #     Args:
    #         idx ([type]): [description]

    #     Returns:
    #         [type]: [description]
    #     """
    #     data = self.data[idx]
    #     seg_data, _ = load(data['seg'])
    #     img_data, _ = load(data['img'])

    #     vtk_data = pyvista.read(data['vtp'])

    #     seg_data = (torch.from_numpy(seg_data).float()/255.).unsqueeze(0)

    #     img_data = np.moveaxis(img_data, 0, -1)
    #     img_data = torch.from_numpy(np.array(Image.fromarray(img_data).convert('L')) / 255.).unsqueeze(0)

    #     img_data = self.resize(img_data)
    #     seg_data = self.resize(seg_data)
    #     seg_data[seg_data < 0.3] = 0
    #     seg_data[seg_data >= 0.3] = 1

    #     if self.real_set_augment:
    #         growth_factor = random.randint(self.growth_lower_bound, self.growth_upper_bound)

    #         x = seg_data.unsqueeze(0).unsqueeze(0).float()
    #         for i in range(growth_factor):
    #             x = conv2d(x, weight=self.growth_tensor, padding=1)

    #         x[x > 1] = 1
    #         x[x < 1] = 0
    #         seg_data = x - 0.5

    #         seg_data = self.resize(seg_data[0])
    #         img_data = seg_data


    #     coordinates = torch.tensor(np.float32(
    #         np.asarray(vtk_data.points)), dtype=torch.float)[:, :2]
    #     lines = torch.tensor(np.asarray(
    #         vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]

    #     z_pos = None

    #     if self.gaussian_augment:
    #         z_pos = random.randint(3, seg_data.shape[-1] - 4) / (seg_data.shape[-1] - 1) if not self.rotate else 0.5

    #         seg_data = self._project_image_3d(seg_data, z_pos)
    #         img_data = self._project_image_3d(img_data, z_pos)

    #         coordinates = F.pad(coordinates, (0, 1), "constant", z_pos)
    #         coordinates = coordinates[:, [1, 0, 2]]

    #         if self.rotate:
    #             if self.continuous:
    #                 alpha = random.randint(0, 270)
    #                 beta = random.randint(0, 270)
    #                 gamma = random.randint(0, 270)
    #             else:
    #                 alpha = random.randint(0, 3) * 90
    #                 beta = random.randint(0, 3) * 90
    #                 gamma = random.randint(0, 3) * 90

    #             seg_data = rotate_image(seg_data, alpha, beta, gamma)
    #             img_data = rotate_image(img_data, alpha, beta, gamma)
    #             coordinates = rotate_coordinates(coordinates, alpha, beta, gamma)

    #             seg_data[seg_data >= 0] = 0.5
    #             seg_data[seg_data < 0] = -0.5

    #         if self.real_set_augment:
    #             img_data = seg_data.clone()
    #             img_data[seg_data > 0] -= 0.5
    #             img_data[seg_data <= 0] += 0.25

    #             img_data = self.gaussian_noise(img_data)

    #             img_data = self.gaussian_smooth(img_data)

    #         img_data = self.scaling(img_data)

    #         # Add padding to the borders
    #         img_data = self.pad_transform(img_data)
    #         seg_data = self.pad_transform(seg_data)

    #     else:
    #         img_data = self.gaussian_noise(img_data)
    #         """image_data[image_data > 0.5] = 0.5
    #         image_data = self.gaussian_smooth(image_data)"""
    #         img_data = self.scaling(img_data)
    #         img_data = self.pad_transform2d(img_data)
    #         seg_data = self.pad_transform2d(seg_data)
    #         coordinates = coordinates[:, [1, 0]]

    #     coordinates = (coordinates * (self.size - 2 * self.padding) + self.padding) / self.size

    #     return [img_data], [seg_data], [coordinates], [lines], [z_pos], [self.domain_classification]
    

    def __getitem__(self, idx):
        data = self.data[idx]
        seg_data, _ = load(data['seg'])
        img_data, _ = load(data['img'])
        vtk_data = pyvista.read(data['vtp'])

        # ---- geometric-only aug params ----
        do_flip = True
        do_elastic_2d = False
        elastic_prob = 0.8
        elastic_sigma_px = 6.0   # smoothing in pixels
        elastic_mag_px = 2.0     # max-ish displacement in pixels (start 1-3)
        apply_scaling = True

        # ---- Load + preprocess 2D ----
        seg_data = self._normalize_loaded_2d(seg_data)
        img_data = self._normalize_loaded_2d(img_data)

        seg_data = (torch.from_numpy(seg_data).float() / 255.).unsqueeze(0)  # (1,H,W)
        img_data = (torch.from_numpy(img_data).float() / 255.).unsqueeze(0)  # (1,H,W)

        img_data = self.resize(img_data)
        seg_data = self.resize(seg_data)
        seg_data[seg_data < 0.3] = 0
        seg_data[seg_data >= 0.3] = 1

        if self.real_set_augment:
            growth_factor = random.randint(self.growth_lower_bound, self.growth_upper_bound)
            x = seg_data.unsqueeze(0).unsqueeze(0).float()
            for _ in range(growth_factor):
                x = conv2d(x, weight=self.growth_tensor, padding=1)
            x[x > 1] = 1
            x[x < 1] = 0
            seg_data = x - 0.5
            seg_data = self.resize(seg_data[0])
            img_data = seg_data

        # ---- Graph data ----
        coordinates = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)[:, :2]  # (N,2) in [0,1]
        lines = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)[:, 1:]

        # =========================
        # 2D ELASTIC (before 3D!)
        # =========================
        def _gaussian_kernel1d(sigma: float, device, dtype):
            radius = max(1, int(3.0 * sigma))
            x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
            k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
            return k / k.sum()

        def _smooth_field_2d(field, sigma: float):
            # field: (1,2,H,W)
            device, dtype = field.device, field.dtype
            k1d = _gaussian_kernel1d(sigma, device, dtype)
            K = k1d.numel()

            w = k1d.view(1, 1, K, 1)
            field = F.conv2d(field, w.repeat(2, 1, 1, 1), padding=(K // 2, 0), groups=2)
            w = k1d.view(1, 1, 1, K)
            field = F.conv2d(field, w.repeat(2, 1, 1, 1), padding=(0, K // 2), groups=2)
            return field

        def _make_base_grid_2d(H, W, device, dtype):
            y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
            x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            return torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1,H,W,2) (x,y)
        
        def _make_base_grid_2d_actrue(H, W, device, dtype):
            """
            Identity grid for grid_sample with align_corners=True.

            Normalized coords in [-1,1] map to pixel centers:
            -1 -> 0
            1 -> W-1 (or H-1)

            This matches the way your original code maps
            normalized coordinates to pixel indices via (W-1).
            """
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            return torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1,H,W,2) (x,y)


        def _elastic_2d(img, seg, coords_xy01, mag_px=2.0, sigma_px=6.0):
            """
            img, seg: (1,H,W) after self.resize (so W=H=54 in your logs)
            coords_xy01: (N,2) stored as (pixel / 128) convention from patch extraction
            Uses align_corners=True but converts coords correctly for your denom=128 convention.
            """
            device, dtype = img.device, img.dtype
            _, H, W = img.shape

            # px -> normalized disp magnitude for align_corners=True
            mag_x = (2.0 * mag_px) / max(1.0, (W - 1))
            mag_y = (2.0 * mag_px) / max(1.0, (H - 1))

            base = _make_base_grid_2d_actrue(H, W, device, dtype)  # linspace(-1,1) grid

            disp = torch.randn((1, 2, H, W), device=device, dtype=dtype)
            disp = _smooth_field_2d(disp, sigma_px)
            disp = disp / (disp.std() + 1e-6)
            disp[:, 0] *= mag_x  # dx in normalized coords
            disp[:, 1] *= mag_y  # dy in normalized coords

            grid = base + disp.permute(0, 2, 3, 1).contiguous()  # (1,H,W,2)

            # Warp image/seg
            img_w = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)[0]
            seg_w = F.grid_sample(seg.unsqueeze(0), grid, mode='nearest',
                                padding_mode='zeros', align_corners=True)[0]

            # --- Node displacement sampling: IMPORTANT PART ---
            coords_xy = coords_xy01.clamp(0, 1)

            # Your coords are "pixel/W0" (W0=128), so after resize to W:
            # pixel coordinate in resized image is x = coords * W (continuous in [0, W])
            # align_corners=True expects normalization via (W-1), so use x/(W-1)
            x_norm = 2.0 * (coords_xy[:, 0] * W / max(1.0, (W - 1))) - 1.0
            y_norm = 2.0 * (coords_xy[:, 1] * H / max(1.0, (H - 1))) - 1.0
            coords_norm = torch.stack([x_norm, y_norm], dim=1)  # (N,2) in [-1,1]

            g = coords_norm.view(1, 1, -1, 2)
            disp_at = F.grid_sample(disp, g, mode='bilinear',
                                    padding_mode='border', align_corners=True)
            disp_at = disp_at.view(1, 2, -1).squeeze(0).transpose(0, 1).contiguous()  # (N,2)

            coords_norm_w = coords_norm + disp_at

            # Convert back: norm -> pixel (0..W-1), then pixel -> coord (divide by W, not W-1)
            x_pix = ((coords_norm_w[:, 0] + 1.0) * 0.5) * (W - 1)
            y_pix = ((coords_norm_w[:, 1] + 1.0) * 0.5) * (H - 1)

            coords_xy_w = torch.stack([x_pix / W, y_pix / H], dim=1).clamp(0, 1)

            return img_w, seg_w, coords_xy_w

        if do_elastic_2d and (random.random() < elastic_prob):
            # coords are currently (N,2) as (x,y) in [0,1]
            coords_xy = coordinates[:, [0, 1]]
            img_data, seg_data, coords_xy_w = _elastic_2d(
                img_data, seg_data, coords_xy,
                mag_px=elastic_mag_px,
                sigma_px=elastic_sigma_px
            )
            coordinates = coords_xy_w  # still (x,y)

            # Hard binarize seg again to avoid interpolation fuzz
            seg_data[seg_data < 0.5] = 0.0
            seg_data[seg_data >= 0.5] = 1.0

            # If you want *zero chance* of image-only junk: tie image to seg
            if self.real_set_augment:
                img_data = seg_data.clone()

        # =========================
        # Now project to 3D (clean)
        # =========================
        z_pos = random.randint(3, seg_data.shape[-1] - 4) / (seg_data.shape[-1] - 1) if not self.rotate else 0.5

        seg_data = self._project_image_3d(seg_data, z_pos)  # (1,H,W,Z)
        img_data = self._project_image_3d(img_data, z_pos)

        # coords -> (y,x,z) with your convention
        coordinates = F.pad(coordinates, (0, 1), "constant", z_pos)  # (N,3) (x,y,z)
        coordinates = coordinates[:, [1, 0, 2]]  # (y,x,z)

        # ---- your existing 3D rotations ----
        if self.rotate:
            if self.continuous:
                alpha = random.randint(0, 270)
                beta = random.randint(0, 270)
                gamma = random.randint(0, 270)
            else:
                alpha = random.randint(0, 3) * 90
                beta = random.randint(0, 3) * 90
                gamma = random.randint(0, 3) * 90

            seg_data = rotate_image(seg_data, alpha, beta, gamma)
            img_data = rotate_image(img_data, alpha, beta, gamma)
            coordinates = rotate_coordinates(coordinates, alpha, beta, gamma)

        # Put seg into your expected range
        seg_data = seg_data.clone()
        seg_data[seg_data >= 0] = 0.5
        seg_data[seg_data < 0] = -0.5

        # ---- post-3D flips (safe) ----
        if do_flip:
            if random.random() < 0.5:  # flip X (W)
                img_data = torch.flip(img_data, dims=[2])
                seg_data = torch.flip(seg_data, dims=[2])
                coordinates[:, 1] = 1.0 - coordinates[:, 1]
            if random.random() < 0.5:  # flip Y (H)
                img_data = torch.flip(img_data, dims=[1])
                seg_data = torch.flip(seg_data, dims=[1])
                coordinates[:, 0] = 1.0 - coordinates[:, 0]
            if random.random() < 0.5:  # flip Z
                img_data = torch.flip(img_data, dims=[3])
                seg_data = torch.flip(seg_data, dims=[3])
                coordinates[:, 2] = 1.0 - coordinates[:, 2]

        if self.real_set_augment:
            # Keep your intensity convention if you want
            img_data = seg_data.clone()
            img_data[seg_data > 0] -= 0.5
            img_data[seg_data <= 0] += 0.25

        if apply_scaling:
            img_data = self.scaling(img_data)

        img_data = self.pad_transform(img_data)
        seg_data = self.pad_transform(seg_data)

        coordinates = (coordinates * (self.size - 2 * self.padding) + self.padding) / self.size

        return [img_data], [seg_data], [coordinates], [lines], [z_pos], [self.domain_classification]



def build_plants_network_data(config, mode='train', split=0.95, debug=False, gaussian_augment=False, rotate=False, continuous=False, max_samples=0, domain_classification=-1):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    if mode == 'train':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        ds = Plants2GraphDataLoader(
            data=data_dicts,
            transform=train_transform,
            gaussian_augment=gaussian_augment,
            rotate=rotate,
            continuous=continuous,
            size=config.DATA.IMG_SIZE[0],
            padding=config.DATA.PAD_SIZE[0],
            real_set_augment=False and (config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels") ,
            growth_range=config.DATA.GROWTH_RANGE
        )
        return ds
    elif mode == 'test':
        img_folder = os.path.join(config.DATA.SOURCE_DATA_PATH, 'test/raw')
        seg_folder = os.path.join(config.DATA.SOURCE_DATA_PATH, 'test/seg')
        vtk_folder = os.path.join(config.DATA.SOURCE_DATA_PATH, 'test/vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
        ds = Plants2GraphDataLoader(
            data=data_dicts,
            transform=val_transform,
            gaussian_augment=gaussian_augment,
            rotate=rotate,
            continuous=continuous,
            size=config.DATA.IMG_SIZE[0],
            padding=config.DATA.PAD_SIZE[0],
            real_set_augment=False and (config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels")  ,
            growth_range=config.DATA.GROWTH_RANGE
        )
        return ds
    elif mode == 'split':
        
        data_root = Path(getattr(config.DATA, "SOURCE_DATA_PATH", None) or config.DATA.DATA_PATH)
        
        # ---- TRAIN FOLDERS ----
        train_root_path = data_root / "train"
        
        img_folder_train = train_root_path / "raw"
        seg_folder_train = train_root_path / "seg"
        vtk_folder_train = train_root_path / "vtp"
        img_files_train = []
        vtk_files_train = []
        seg_files_train = []

        for file_ in os.listdir(img_folder_train):
            file_ = file_[:-8]
            img_files_train.append(os.path.join(str(img_folder_train), file_+'data.png'))
            vtk_files_train.append(os.path.join(str(vtk_folder_train), file_+'graph.vtp'))
            seg_files_train.append(os.path.join(str(seg_folder_train), file_+'seg.png'))

        data_dicts_train = [
            {"img": img_file_train, "vtp": vtk_file_train, "seg": seg_file_train} for img_file_train, vtk_file_train, seg_file_train in zip(img_files_train, vtk_files_train, seg_files_train)
        ]
        
        # ---- VAL FOLDERS ----
        val_root = data_root / "val"
        
        img_folder_val = val_root / "raw"
        seg_folder_val = val_root / "seg"
        vtk_folder_val = val_root / "vtp"
        img_files_val = []
        vtk_files_val = []
        seg_files_val = []

        for file_ in os.listdir(img_folder_val):
            file_ = file_[:-8]
            img_files_val.append(os.path.join(str(img_folder_val), file_+'data.png'))
            vtk_files_val.append(os.path.join(str(vtk_folder_val), file_+'graph.vtp'))
            seg_files_val.append(os.path.join(str(seg_folder_val), file_+'seg.png'))

        data_dicts_val = [
            {"img": img_file_val, "vtp": vtk_file_val, "seg": seg_file_val} for img_file_val, vtk_file_val, seg_file_val in zip(img_files_val, vtk_files_val, seg_files_val)
        ]
        
        train_files = data_dicts_train
        val_files = data_dicts_val

        if debug:
            train_files = train_files[:128]
            val_files = train_files[:32]
        elif max_samples > 0:
            train_files = train_files[:max_samples]
            val_files = val_files[:min(round(max_samples*(1-split)), config.DATA.BATCH_SIZE * 10)]

        train_ds = Plants2GraphDataLoader(
            data=train_files,
            transform=train_transform,
            gaussian_augment=gaussian_augment,
            rotate=rotate,
            continuous=continuous,
            size=config.DATA.IMG_SIZE[0],
            padding=config.DATA.PAD_SIZE[0],
            real_set_augment=False and (config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels") ,
            growth_range=config.DATA.GROWTH_RANGE,
            domain_classification=domain_classification
        )
        val_ds = Plants2GraphDataLoader(
            data=val_files,
            transform=val_transform,
            gaussian_augment=gaussian_augment,
            rotate=rotate,
            continuous=continuous,
            size=config.DATA.IMG_SIZE[0],
            padding=config.DATA.PAD_SIZE[0],
            real_set_augment=False and (config.DATA.DATASET == "real_vessels" or config.DATA.DATASET == "mixed_real_vessels") ,
            growth_range=config.DATA.GROWTH_RANGE,
            domain_classification=domain_classification
        )
        return train_ds, val_ds, None
