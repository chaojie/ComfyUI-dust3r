import os
import sys
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
sys.path.append(f'{comfy_path}/custom_nodes/ComfyUI-dust3r')
#print(sys.path)

from PIL import Image

import argparse
import torch
import numpy as np
import tempfile
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation

from .dust3r.inference import inference, load_model
from .dust3r.image_pairs import make_pairs
from .dust3r.utils.image import load_images, rgb
from .dust3r.utils.device import to_numpy
from .dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from .dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

pretrained_weights_path=f'{comfy_path}/custom_nodes/ComfyUI-dust3r/checkpoints'
pretrained_weights=os.listdir(pretrained_weights_path)

input_path=f'{comfy_path}/custom_nodes/ComfyUI-dust3r/input'
output_path=f'{comfy_path}/custom_nodes/ComfyUI-dust3r/output'

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False, transparent_cams=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size)


def get_reconstructed_scene(outdir, model, device, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile = get_3D_model_from_scene(outdir, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d/depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs
    
class Dust3rLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (pretrained_weights, {"default": "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"}),
                "device": ("STRING",{"default":"cuda"}),
            },
        }

    RETURN_TYPES = ("Dust3rModel",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "Dust3r"

    def run(self,path,device):
        path=f'{pretrained_weights_path}/{path}'
        model = load_model(path, device)

        return (model,)

class Dust3rRun:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("Dust3rModel",),
                "device": ("STRING",{"default":"cuda"}),
                "images": ("IMAGE",),
                "image_size": ("INT",{"default":512}),
                "scenegraph_type": (["complete","swin","oneref"],{"default":"complete"}),
                "schedule": (["linear", "cosine"],{"default":"linear"}),
                "niter": ("INT", {"default": 300, "min": 0, "max": 5000}),
                "min_conf_thr": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20, "step": 0.1}),
                "cam_size": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.1, "step": 0.001}),
                "as_pointcloud": ("BOOLEAN", {"default": False}),
                "mask_sky": ("BOOLEAN", {"default": False}),
                "clean_depth": ("BOOLEAN", {"default": True}),
                "transparent_cams": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "Dust3r"

    def run(self,model,device,images,image_size,scenegraph_type,schedule,niter,min_conf_thr,cam_size,as_pointcloud,mask_sky,clean_depth,transparent_cams):
        winsize=1
        refid=0
        num_files = len(images)
        max_winsize = max(1, (num_files - 1)//2)
        
        
        if scenegraph_type == "swin":
            winsize = max_winsize
            refid = 0
        elif scenegraph_type == "oneref":
            winsize = max_winsize
            refid = 0
        else:
            winsize = max_winsize
            refid = 0
        
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            os.remove(file_path)
        ind = 0
        for image in images:
            image = 255.0 * image.cpu().numpy()
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            image_path=f'{input_path}/{ind}.png'
            image.save(image_path)
            ind=ind+1
        imgs = load_images(input_path, size=image_size)
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
        if scenegraph_type == "swin":
            scenegraph_type = scenegraph_type + "-" + str(winsize)
        elif scenegraph_type == "oneref":
            scenegraph_type = scenegraph_type + "-" + str(refid)
        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)
        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode)
        lr = 0.01
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

        outfile = get_3D_model_from_scene(output_path, scene, min_conf_thr, as_pointcloud, mask_sky,
                                          clean_depth, transparent_cams, cam_size)

        # also return rgb, depth and confidence imgs
        # depth is normalized with the max value for all images
        # we apply the jet colormap on the confidence maps
        rgbimg = scene.imgs
        depths = to_numpy(scene.get_depthmaps())
        confs = to_numpy([c for c in scene.im_conf])
        cmap = pl.get_cmap('jet')
        depths_max = max([d.max() for d in depths])
        depths = [d/depths_max for d in depths]
        confs_max = max([d.max() for d in confs])
        confs = [cmap(d/confs_max) for d in confs]

        imgs = []
        for i in range(len(rgbimg)):
            imgs.append(rgbimg[i])
            imgs.append(rgb(depths[i]))
            imgs.append(rgb(confs[i]))

        return (outfile,)

NODE_CLASS_MAPPINGS = {
    "Dust3rLoader":Dust3rLoader,
    "Dust3rRun":Dust3rRun,
}
