import torch
from torch import nn as nn
import torchvision

from ..utils.transformations import transform_points_Rt
from .alignment import align
from .backbones import ResNetDecoder, ResNetEncoder
from .correspondence import get_correspondences
from .model_util import get_grid, grid_to_pointcloud, points_to_ndc
from .renderer import PointsRenderer


_DEBUG_MODE_ENABLED = False


def create_img_pair_visual(
    image1,
    image2,
    img_height,
    img_width,
    matched_keypoints,
    matched_warped_keypoints,
):
    import cv2
    import numpy as np

    # load in images of shape (img_height, img_width)
    image1 = image1.permute((1, 2, 0))
    image1 = image1.squeeze()
    image1 = image1.detach().cpu().numpy()
    image1 = (image1 * 255.0).astype(np.uint8)

    image2 = image2.permute((1, 2, 0))
    image2 = image2.squeeze()
    image2 = image2.detach().cpu().numpy()
    image2 = (image2 * 255.0).astype(np.uint8)

    image_pair = np.hstack((image1, image2))

    # convert keypoints to col, row (x, y) order
    matched_keypoints = matched_keypoints[:, [1, 0]]
    matched_warped_keypoints = matched_warped_keypoints[:, [1, 0]]

    matched_keypoints = matched_keypoints.detach().cpu().numpy().astype(int)
    matched_warped_keypoints = (
        matched_warped_keypoints.detach().cpu().numpy().astype(int)
    )

    # draw matched keypoint points and lines associating matched keypoints (point correspondences)
    for i in range(len(matched_keypoints)):
        img1_coords = matched_keypoints[i]
        img2_coords = matched_warped_keypoints[i]
        # add the width so the coordinates show up correctly on the second image
        img2_coords = (img2_coords[0] + img_width, img2_coords[1])

        radius = 1
        thickness = 2
        # points will be red (BGR color)
        image_pair = cv2.circle(
            image_pair, tuple(img1_coords), radius, (0, 0, 255), thickness
        )
        image_pair = cv2.circle(
            image_pair, tuple(img2_coords), radius, (0, 0, 255), thickness
        )

        thickness = 1
        # lines will be green (BGR color)
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        image_pair = cv2.line(
            image_pair, tuple(img1_coords), tuple(img2_coords), color, thickness
        )

    cv2.imwrite("./debug/viz.png", image_pair)

    return image_pair


def draw_interest_points(img, points, color=(255, 0, 0)):
    """Convert img in RGB and draw in red the interest points"""
    import numpy as np

    # handles different shapes of image input (rgb, grayscale with or without channel dim)
    if len(img.shape) == 2:
        img_rgb = np.stack([img, img, img], axis=2)
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            img_rgb = np.concatenate([img, img, img], axis=2)
        elif img.shape[0] == 1:
            img_rgb = np.concatenate([img, img, img], axis=0)
            img_rgb = img_rgb.transpose([1, 2, 0])
        else:
            img_rgb = img
    else:
        raise RuntimeError(f"invalid image shape {img.shape}")

    for i in range(points.shape[0]):
        img_rgb[int(points[i][0]), int(points[i][1]), 0] = color[0]
        img_rgb[int(points[i][0]), int(points[i][1]), 1] = color[1]
        img_rgb[int(points[i][0]), int(points[i][1]), 2] = color[2]
    return img_rgb


def _debug_dump_images(images, positions, prefix):
    device = str(images.device)

    def _dump_images_with_positions(image, path, positions):
        from os import makedirs

        import numpy as np
        from skimage.io import imsave

        makedirs("./debug", exist_ok=True)

        image = image.permute((1, 2, 0))
        image = image.squeeze()
        image = image.detach().cpu().numpy()
        image = (image * 255.0).astype(np.uint8)

        image = draw_interest_points(
            image,
            positions.detach().cpu().numpy(),
        )

        # LOG.warning(f"debug dump image to : {path}")
        print(path)
        imsave(path, image)

    for kk in range(len(images)):
        _dump_images_with_positions(
            images[kk], f"./debug/{prefix}.{device}-img-{kk}.png", positions[kk]
        )
    print(f'debug mode enabled on "{__file__}"')


def project_rgb(pc_0in1_X, rgb_src, renderer):
    # create rgb_features
    B, _, H, W = rgb_src.shape
    rgb_src = rgb_src.view(B, 3, H * W)
    rgb_src = rgb_src.permute(0, 2, 1).contiguous()

    # Rasterize and Blend
    project_0in1 = renderer(pc_0in1_X, rgb_src)

    return project_0in1["feats"]


class PCReg(nn.Module):
    def __init__(self, cfg, encoder=None):
        super(PCReg, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        if encoder is None:
            self.encode = ResNetEncoder(chan_in, feat_dim, pretrained)
        else:
            self.encode = encoder

        if encoder is None:
            self.decode = ResNetDecoder(feat_dim, 3, nn.Tanh(), pretrained)
        else:
            self.decode = None

        self.renderer = PointsRenderer(cfg.renderer)
        self.num_corres = cfg.alignment.num_correspodances
        self.pointcloud_source = cfg.renderer.pointcloud_source
        self.align_cfg = cfg.alignment

    def forward(self, rgbs, K, deps, vps=None, padding=0):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
            H == 128 and W == 128
        ), "feature size should be equal to 128, please change the input size accordingly (using `--img_dim`)"

        if feats[0].shape[-1] != deps[0].shape[-1]:
            padding = int((deps[0].shape[-1] - feats[0].shape[-1]) / 2)
            assert len(deps) == 2
            deps = [
                dep[..., padding:-padding, padding:-padding].contiguous()
                for dep in deps
            ]
            rgbs = [
                rgb[..., padding:-padding, padding:-padding].contiguous()
                for rgb in rgbs
            ]
        else:
            padding = 0

        output["padding"] = padding

        grid = get_grid(B, H, W)
        grid[:, :2] += padding
        grid = grid.to(deps[0])

        K_inv = K.inverse()

        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid)
            for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )

                if _DEBUG_MODE_ENABLED:
                    # for j in range(B):
                    _debug_dump_images(
                        rgbs[0],
                        grid[0]
                        .view(3, 128 * 128)
                        .permute(1, 0)[..., [1, 0]][corr_i[0].squeeze(0)]
                        .unsqueeze(0),
                        "0",
                    )
                    _debug_dump_images(
                        rgbs[i],
                        grid[0]
                        .view(3, 128 * 128)
                        .permute(1, 0)[..., [1, 0]][corr_i[1].squeeze(0)]
                        .unsqueeze(0),
                        "1",
                    )
                    create_img_pair_visual(
                        rgbs[0][0][:1],
                        rgbs[i][0][:1],
                        rgbs[0].shape[2],
                        rgbs[0].shape[3],
                        grid.view(3, 128 * 128).permute(1, 0)[..., [1, 0]][
                            corr_i[0].squeeze(0)
                        ],
                        grid.view(3, 128 * 128).permute(1, 0)[..., [1, 0]][
                            corr_i[1].squeeze(0)
                        ],
                    )

                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1 : n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1 : n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1 : n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            if self.decode:
                output[f"rgb_decode_{i}"] = self.decode(proj_F_i)

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def forward_pcreg(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        vps = []
        cor_loss = []
        for i in range(1, n_views):
            corr_i = get_correspondences(
                P1=pcs_F[0],
                P2=pcs_F[i],
                P1_X=pcs_X[0],
                P2_X=pcs_X[i],
                num_corres=self.num_corres,
                ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
            )
            Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

            vps.append(Rt_i)
            cor_loss.append(cor_loss_i)

            # add for visualization
            output[f"corres_0{i}"] = corr_i
            output[f"vp_{i}"] = Rt_i

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        return output

    def generate_pointclouds(self, K, deps, vps=None, padding=0):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        if padding > 0:
            grid[:, :2] += padding
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(
                    pcs_X[i + 1],
                    vps[i + 1],
                    inverse=True,
                )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
            feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


class LoFTR(nn.Module):
    def __init__(self, cfg):
        super(LoFTR, self).__init__()

        self.cfg = cfg
        self.num_corres = cfg.alignment.num_correspodances
        self.pointcloud_source = cfg.renderer.pointcloud_source
        self.align_cfg = cfg.alignment

        # load loftr package
        import sys

        loftr_path = "/private/home/gleize/code/external/LoFTR"
        sys.path.append(loftr_path)

        from src.lightning.lightning_loftr import PL_LoFTR
        from configs.loftr.indoor import loftr_ds as conf

        # to produce dense matches
        conf.cfg.LOFTR.MATCH_COARSE.THR = 0.0

        # ckpt = f"{loftr_path}/weights/indoor_ds_new.ckpt"
        ckpt = f"{loftr_path}/weights/outdoor_ds.ckpt"

        conf.cfg["LOFTR"]["COARSE"]["TEMP_BUG_FIX"] = ckpt.endswith(
            "indoor_ds_new.ckpt"
        )  # set to False when using the old ckpt

        self.model = PL_LoFTR(
            conf.cfg,
            pretrained_ckpt=ckpt,
            dump_dir=None,
        )
        self.model.eval()

    def forward(self, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        assert n_views == 2
        output = {}

        image0 = torchvision.transforms.functional.rgb_to_grayscale(rgbs[0])
        image1 = torchvision.transforms.functional.rgb_to_grayscale(rgbs[1])

        loftr_data = {
            "image0": image0,
            "image1": image1,
        }
        self.model.matcher(loftr_data)

        B, _, H, W = image0.shape

        pts0 = loftr_data["mkpts0_f"]
        pts1 = loftr_data["mkpts1_f"]
        b_ids = loftr_data["b_ids"]
        i_ids = loftr_data["i_ids"]
        j_ids = loftr_data["j_ids"]
        mconf = loftr_data["mconf"]

        if _DEBUG_MODE_ENABLED:
            _debug_dump_images(
                loftr_data["image0"],
                loftr_data["mkpts0_f"][:, [1, 0]].unsqueeze(0),
                "0",
            )
            _debug_dump_images(
                loftr_data["image1"],
                loftr_data["mkpts1_f"][:, [1, 0]].unsqueeze(0),
                "1",
            )
            create_img_pair_visual(
                loftr_data["image0"][0],
                loftr_data["image1"][0],
                loftr_data["image0"].shape[2],
                loftr_data["image0"].shape[3],
                loftr_data["mkpts0_f"][:, [1, 0]],
                loftr_data["mkpts1_f"][:, [1, 0]],
            )

        _1 = torch.ones((pts0.shape[0], 1), device=pts0.device)
        pts0 = torch.cat((pts0, _1), dim=1)
        pts1 = torch.cat((pts1, _1), dim=1)

        K_inv = K.inverse()

        pointclouds = [
            grid_to_pointcloud(K_inv, deps[0], None, pts0, b_ids=b_ids),
            grid_to_pointcloud(K_inv, deps[1], None, pts1, b_ids=b_ids),
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        
        vps = []
        cor_loss = []
        for i in range(1, n_views):
            Rt_i = []
            cor_loss_i = []
            for b in range(B):
                mask = b_ids == b
                corr_i = (
                    i_ids[mask].unsqueeze(0),
                    j_ids[mask].unsqueeze(0),
                    mconf[mask].unsqueeze(0),
                    None,
                )
                pX0 = pcs_X[0][mask].unsqueeze(0)
                pX1 = pcs_X[i][mask].unsqueeze(0)

                Rt_i_, cor_loss_i_ = align(corr_i, pX0, pX1, self.align_cfg)

                Rt_i.append(Rt_i_)
                cor_loss_i.append(cor_loss_i_)

            Rt_i = torch.cat(Rt_i)
            cor_loss_i = torch.cat(cor_loss_i)

            vps.append(Rt_i)
            cor_loss.append(cor_loss_i)

            # add for visualization
            output[f"vp_{i}"] = Rt_i

        pointclouds = [
            grid_to_pointcloud(K_inv, deps[0], None, None),
            grid_to_pointcloud(K_inv, deps[1], None, None),
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        output["padding"] = 0

        return output

    def generate_pointclouds(self, K, deps, vps=None, padding=0):
        assert padding == 0

        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(
                    pcs_X[i + 1],
                    vps[i + 1],
                    inverse=True,
                )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X
