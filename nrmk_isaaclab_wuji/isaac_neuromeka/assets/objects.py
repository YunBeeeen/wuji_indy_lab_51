import pickle
from collections.abc import Sequence

import torch
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.utils.math import matrix_from_quat


class RigidObject_w_FullPCL(RigidObject):
    def __init__(self, cfg):
        super().__init__(cfg)

        # load pre-computed pcl
        self.env_id_to_obj_path = dict()
        with h5py.File("env_id_to_obj_path.h5", "r") as root:
            for k in root.keys():
                self.env_id_to_obj_path[int(k)] = root[k][()].decode("utf-8")
        self.num_pcl = 200  # TODO: Hard-code
        self.nominal_pcl = None

    def _create_buffers(self):
        super()._create_buffers()

        # load nominal pcl
        if self.nominal_pcl is None:
            self.nominal_pcl = torch.zeros(self.num_instances, self.num_pcl, 3, device=self.device)
            for env_id, usd_path in self.env_id_to_obj_path.items():
                pcl_path = "/".join(usd_path.split("/")[:-2]) + f"/point_cloud_{self.num_pcl}_pts.pkl"
                with open(pcl_path, "rb") as f:
                    self.nominal_pcl[env_id] = torch.from_numpy(pickle.load(f))  # (500, 3)

        self.full_pcl = self.nominal_pcl

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        super().write_root_state_to_sim(root_state, env_ids)

        if env_ids is None:
            env_ids = slice(None)
        root_quat = self._data.root_state_w[env_ids, 3:7]  # (w, x, y, z)
        root_rotMat = matrix_from_quat(root_quat)  # (len(env_ids), 3, 3))
        root_rotMat_ = root_rotMat.unsqueeze(1)  # (len(env_ids), 1, 3, 3)
        nominal_pcl_ = self.nominal_pcl[env_ids].unsqueeze(-1)  # (len(env_ids), 500, 3, 1)
        self.full_pcl[env_ids] = torch.matmul(root_rotMat_, nominal_pcl_).squeeze(-1)  # (len(env_ids), 500, 3)

    def update(self, dt: float):
        super().update(dt)
        root_quat = self._data.root_state_w[:, 3:7]
        root_rotMat = matrix_from_quat(root_quat)
        root_rotMat_ = root_rotMat.unsqueeze(1)  # (len(env_ids), 1, 3, 3)
        nominal_pcl_ = self.nominal_pcl.unsqueeze(-1)  # (len(env_ids), 500, 3, 1)
        self.full_pcl = torch.matmul(root_rotMat_, nominal_pcl_).squeeze(-1)  # (len(env_ids), 500, 3)

