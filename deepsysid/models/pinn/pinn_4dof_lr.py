import torch.nn as nn
import torch
from torch.utils import data
from typing import Dict, List, Literal, Optional, Tuple
from numpy.typing import NDArray
import numpy as np



class PINNNet(nn.Module):
    def __init__(self, inputNode=7, hiddenNode=256, outputNode=4):
        super(PINNNet, self).__init__()
        # Define Hyperparameters
        self.inputLayerSize = inputNode
        self.outputLayerSize = outputNode
        self.hiddenLayerSize = hiddenNode
        # weights
        self.Linear1 = nn.Linear(self.inputLayerSize, self.hiddenLayerSize)
        self.Linear2 = nn.Linear(self.hiddenLayerSize, self.outputLayerSize)
        self.activation = torch.nn.Sigmoid()

        # Define Frigate Hyperparameters
        g = 9.81
        m = 365.79 * pow(10, 3)
        self.Xudot = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Xuau = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Xvr = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Yvdot = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Ypdot = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Yrdot = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Yauv = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Yur = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Yvav = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Yvar = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Yrav = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Ybauv = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Ybaur = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)
        self.Ybuu = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0), )*m*g, requires_grad=True)

        self.Kvdot = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kpdot= nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Krdot = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kauv = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kur = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kvav = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kvar = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Krav = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kbauv = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kbaur = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kbuu = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kaup = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kpap = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kp = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kb = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Kbbb = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)

        self.Nvdot= nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Npdot = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Nrdot = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Nauv= nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Naur = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Nrar = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Nrav = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Nbauv= nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Nbuar = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)
        self.Nbuau = nn.Parameter(torch.normal(torch.tensor(0.0), torch.tensor(1.0)*m*g, ), requires_grad=True)



    def forward(self, X):
        out1 = self.Linear1(X)
        out2 = self.activation(out1)
        out3 = self.Linear2(out2)
        return out3

    def pinn_loss_4dof(self, u, v, p, r, phi, u_prev, v_prev, p_prev, r_prev):
        u = u.detach().cpu().numpy().astype(float)
        v = v.detach().cpu().numpy().astype(float)
        p = p.detach().cpu().numpy().astype(float)
        r = r.detach().cpu().numpy().astype(float)
        phi = phi.detach().cpu().numpy().astype(float)
        u_prev = u_prev.detach().cpu().numpy().astype(float)
        v_prev = v_prev.detach().cpu().numpy().astype(float)
        p_prev = p_prev.detach().cpu().numpy().astype(float)
        r_prev = r_prev.detach().cpu().numpy().astype(float)
        # Constant
        rho_water = 1025.0
        g = 9.81

        # Main Particulars
        Lpp = 51.5
        B = 8.6
        D = 2.3

        # Load condition
        disp = 355.88
        # m = 365.79 * 10 ^ 3
        m = 365.79 * pow(10, 3)
        Izz = 3.3818 * pow(10, 7)
        Ixx = 3.4263 * pow(10, 6)
        gm = 1.0
        LCG = 20.41
        VCG = 3.36
        xG = -3.38
        zG = -1.06
        normalize = g * m

        # # Normalization
        # self.Xudot = torch.float(self.Xudot) * g * m
        # self.Xuau = torch.float(self.Xuau) * g * m
        # self.Xvr = torch.float(self.Xvr) * g * m
        # self.Yvdot = torch.float(self.Yvdot) * g * m
        # self.Ypdot = torch.float(self.Ypdot) * g * m
        # self.Yrdot = torch.float(self.Yrdot) * g * m
        # self.Yauv = torch.float(self.Yauv) * g * m
        # self.Yur = torch.float(self.Yur) * g * m
        # self.Yvav = torch.float(self.Yvav) * g * m
        # self.Yvar = torch.float(self.Yvar) * g * m
        # self.Yrav = torch.float(self.Yrav) * g * m
        # self.Ybauv = torch.float(self.Ybauv) * g * m
        # self.Ybaur = torch.float(self.Ybaur) * g * m
        # self.Ybuu = torch.float(self.Ybuu) * g * m
        #
        # self.Kvdot = torch.float(self.Kvdot) * g * m
        # self.Kpdot= torch.float(self.Kpdot) * g * m
        # self.Krdot = torch.float(self.Krdot ) * g * m
        # self.Kauv = torch.float(self.Kauv) * g * m
        # self.Kur = torch.float(self.Kur) * g * m
        # self.Kvav = torch.float(self.Kvav ) * g * m
        # self.Kvar = torch.float(self.Kvar) * g * m
        # self.Krav = torch.float(self.Krav) * g * m
        # self.Kbauv = torch.float(self.Kbauv) * g * m
        # self.Kbaur = torch.float(self.Kbaur) * g * m
        # self.Kbuu = torch.float(self.Kbuu) * g * m
        # self.Kaup = torch.float(self.Kaup) * g * m
        # self.Kpap = torch.float(self.Kpap) * g * m
        # self.Kp = torch.float(self.Kp) * g * m
        # self.Kb = torch.float(self.Kb) * g * m
        # self.Kbbb = torch.float(self.Kbbb) * g * m
        #
        # self.Nvdot= torch.float(self.Nvdot) * g * m
        # self.Npdot = torch.float(self.Npdot) * g * m
        # self.Nrdot = torch.float(self.Nrdot) * g * m
        # self.Nauv= torch.float(self.Nauv) * g * m
        # self.Naur = torch.float(self.Naur) * g * m
        # self.Nrar =torch.float(self.Nrar) * g * m
        # self.Nrav = torch.float(self.Nrav) * g * m
        # self.Nbauv= torch.float(self.Nbauv) * g * m
        # self.Nbuar = torch.float(self.Nbuar) * g * m
        # self.Nbuau = torch.float(self.Nbuau) * g * m


        # Auxiliary variables
        b = phi
        au = abs(u)
        av = abs(v)
        ar = abs(r)
        ap = abs(p)

        # Total Mass Matrix
        M = torch.from_numpy(np.array([
            [(m - self.Xudot), 0, 0, 0],
            [0, (m - self.Yvdot), -(m * zG + self.Ypdot), (m * xG - self.Yrdot)],
            [0, -(m * zG + self.Kvdot), (Ixx - self.Kpdot), -self.Krdot],
            [0, (m * xG - self.Nvdot), -self.Npdot, (Izz - self.Nrdot)]
        ]))

        # Hydrodynamic forces without added mass terms (considered in the M matrix)
        Xh = self.Xuau * u * au + self.Xvr * v * r
        # print("Xh is ", Xh)
        Yh = (self.Yauv * au * v + self.Yur * u * r + self.Yvav * v * av + self.Yvar * v * ar + self.Yrav * r * av
              + self.Ybauv * b * abs(u * v) + self.Ybaur * b * abs(u * r) + self.Ybuu * b * pow(u, 2))

        Kh = self.Kauv * au * v + self.Kur * u * r + self.Kvav * v * av + self.Kvar * v * ar + self.Krav * r * av
        + self.Kbauv * b * abs(u * v) + self.Kbaur * b * abs(u * r) + self.Kbuu * b * pow(u, 2) + self.Kaup * au * p
        + self.Kpap * p * ap + self.Kp * p + self.Kbbb * pow(b, 3) - (rho_water * g * gm * disp) * b
        + self.Kb * b

        Nh = self.Nauv * au * v + self.Naur * au * r + self.Nrar * r * ar + self.Nrav * r * av
        + self.Nbauv * b * abs(u * v) + self.Nbuar * b * u * ar + self.Nbuau * b * u * au

        # Rigid - body centripetal accelerations
        Xc = m * (r * v + xG * pow(r, 2) - zG * p * r)
        # print("Xc is ", Xc)
        Yc = - m * u * r
        Kc = m * zG * u * r
        Nc = - m * xG * u * r

        # Total forces
        Xe = 0
        Ye = 0
        Ke = 0
        Ne = 0
        F1 = Xh + Xc + Xe
        F2 = Yh + Yc + Ye
        F4 = Kh + Kc + Ke
        F6 = Nh + Nc + Ne

        F_pred = torch.from_numpy(np.array([F1, F2, F4, F6]))
        # print("F_pred shape is ", F_pred.shape)  # [4, 128, 50])
        sampling_time = 1
        acceleration_true = torch.from_numpy(np.array(
            [(u - u_prev) / sampling_time, (v - v_prev) / sampling_time, (p - p_prev) / sampling_time,
             (r - r_prev) / sampling_time]))
        # print("acceleration_true shape is ", acceleration_true.shape)  # (4, 128, 50)
        # print("M shape is ", M.shape)  # （4x4）
        acceleration_true = acceleration_true.permute(1, 2, 0).unsqueeze(
            2)  # change dim from (4, 128, 50) to (128, 50, 1, 4)
        # print("acceleration_true shape is ", acceleration_true.shape)
        F_true = torch.matmul(acceleration_true, M, out=None)
        # print("F true shape is ", F_true.shape)
        F_true = F_true.squeeze(2).permute(2, 0, 1)
        force_residual = F_true - F_pred
        MSE_R = torch.sum(force_residual * force_residual)
        return MSE_R



class RecurrentPINNDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
            self,
            control_seqs: List[NDArray[np.float64]],
            state_seqs: List[NDArray[np.float64]],
            sequence_length: int,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
            self,
            control_seqs: List[NDArray[np.float64]],
            state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - self.sequence_length - 1) / self.sequence_length
            )

            x = np.zeros(
                (n_samples, self.sequence_length, self.control_dim),
                dtype=np.float64,
            )
            y = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x[idx, :, :] = control[time: time + self.sequence_length, :]
                y[idx, :, :] = state[time: time + self.sequence_length, :]

            x_seq.append(x)
            y_seq.append(y)

        return np.vstack(x_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {'x': self.x[idx], 'y': self.y[idx]}

