import numpy as np
from numpy.random import f
import pinocchio as pin
import logging_mp
logger_mp = logging_mp.getLogger(__name__)
class GravityFeedforward:
    def __init__(self, urdf_path, joint_names=None, root_joint=None):
        """
        urdf_path : str
        joint_names : list[str], optional
        root_joint : pinocchio.JointModel, optional
        """
        if root_joint is None:
            self.model = pin.buildModelFromUrdf(urdf_path)
        else:
            self.model = pin.buildModelFromUrdf(urdf_path, root_joint)

        self.data = self.model.createData()

        if joint_names is None:
            self.joint_id_list = [
                jid for jid in range(1, self.model.njoints)
                if self.model.joints[jid].nq > 0
            ]
        else:
            self.joint_id_list = []
            for name in joint_names:
                jid = self.model.getJointId(name)
                if jid == 0:
                    raise ValueError(f"Joint name {name} not found in URDF.")
                self.joint_id_list.append(jid)
        self.idxs = []
        for jid in self.joint_id_list:
            idx_q = self.model.joints[jid].idx_q
            self.idxs.append(idx_q)
        self.idxs = np.array(self.idxs)
        self.n = len(self.idxs)
    # ------------------------------------------------------------------
    def compute(self, q):
        """
        q : ndarray [n,]
        return:
            tau_ff : ndarray [n,]
        """
        full_q = np.zeros(self.model.nq)
        full_q[self.idxs] = q
        G_full = pin.rnea(
            self.model,
            self.data,
            full_q,
            np.zeros(self.model.nv),
            np.zeros(self.model.nv)
        )

        G_full = np.array(G_full).reshape(-1)


        tau_ff = G_full[self.idxs]
        return tau_ff
