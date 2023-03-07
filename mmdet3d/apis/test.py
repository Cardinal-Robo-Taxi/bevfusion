import mmcv
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

from mmdet3d.core.bbox.structures.utils import get_box_type

def single_gpu_video_test(model, data_loader):
    transform = transforms.ToTensor()
    box_type_3d, box_mode_3d = get_box_type('camera')

    data_loader = [data_loader[0], ]

    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    with torch.no_grad():
        for data in data_loader:
            img = data['frame_center']
            # img = cv2.resize(img, (900, 1600))
            # img = cv2.resize(img, (366, 489))
            img = cv2.resize(img, (512, 352))
            # img = cv2.resize(img, (0, 0), fx=0.7625, fy=0.7625)
            
            img_tensor = transform(img).unsqueeze(0).unsqueeze(0)

            camera_intrinsics = torch.tensor(10**3 * np.array([
                [2.1293,         0,    0.5303],
                [0,         1.9322,    0.2166],
                [0,              0,    0.0010],
            ]).reshape((1,1,3,3)))
            img_aug_matrix = torch.tensor(np.eye(4,4).reshape((1,1,4,4)))
            camera2lidar  = torch.tensor(np.eye(4,4).reshape((1,1,4,4)))
            lidar_aug_matrix  = torch.tensor(np.eye(4,4).reshape((1,1,4,4)))
            metas = [
                dict(
                    box_type_3d=box_type_3d
                )
            ]
            

            print('img_tensor.shape', img_tensor.shape)
            result = model(return_loss=False, rescale=True, img=img_tensor,
                points=None,
                camera2ego=None,
                lidar2ego=None,
                lidar2camera=None,
                lidar2image=None,
                camera_intrinsics=camera_intrinsics,
                camera2lidar=camera2lidar,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=lidar_aug_matrix,
                metas=metas,
            )
        print("result")
        print(result)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
