import torch
import torch.nn as nn

# threshold function for p_g
class my_gen_pred_binary(nn.Module):
    def __init__(self):
        super(my_gen_pred_binary, self).__init__()

    def forward(self, x, thresh):
        right_thresh = thresh.clone().detach().requires_grad_(True).double()
        left_thresh = torch.tensor(0).double()
        x = x.double()
        output = torch.sigmoid(left_thresh - x)
        output = torch.mul(output, left_thresh - x) + x
        output = torch.sigmoid(output - right_thresh)
        output = torch.mul(output, output - right_thresh) + right_thresh
        return output
  
class LossFunc:
    def __init__(self, s_max, G_line, B_line, B_shunt, Br_inv, a, line_loc, device, n_bus):
        self.s = s_max
        self.g = G_line
        self.b = B_line
        self.c = B_shunt
        self.r = Br_inv
        self.a = a
        self.mse = nn.MSELoss()  # MSE loss
        self.lmda1 = torch.tensor(10).to(device)  # V MSE
        self.lmda2 = torch.tensor(1).to(device)  # pi MSE
        self.lmda3 = torch.tensor(0.1).to(device)  # v l_inf
        self.lmda4 = torch.tensor(0.1).to(device)  # s feasibility
        self.lmda5 = torch.tensor(0.01).to(device)  # pi l_inf
        self.line_loc = line_loc
        self.binary_cell = my_gen_pred_binary()
        self.device = device  # 存儲 device
        self.n_bus = n_bus  # 存儲 n_bus

    def calc(self, pred, label, x, feas):
        # 使用 self.n_bus
        mse_p = self.mse(pred[:, :self.n_bus], label[:, :self.n_bus])
        mse_v = self.mse(pred[:, self.n_bus:], label[:, self.n_bus:])
        linf_p = (pred[:, :self.n_bus] - label[:, :self.n_bus]).norm(p=float('inf'))
        linf_v = (pred[:, self.n_bus:] - label[:, self.n_bus:]).norm(p=float('inf'))

        if feas == False:
            return self.lmda1 * mse_v + self.lmda2 * mse_p + self.lmda3 * linf_v + self.lmda5 * linf_p
        
        label_pred = pred[:, :self.n_bus]
        p_max = x[:, :self.n_bus] - x[:, self.n_bus:self.n_bus * 2]
        quadratic_b = x[:, self.n_bus * 4:self.n_bus * 5]
        quadratic_a = x[:, self.n_bus * 5:self.n_bus * 6]
        quadratic_center = (label_pred - quadratic_b) / (quadratic_a + 1e-5) / 2
        p_inj = self.binary_cell(quadratic_center, p_max)
        bus_inj = p_inj + x[:, self.n_bus:self.n_bus * 2]
        p_inj_r = torch.cat((bus_inj[:, :68], bus_inj[:, 69:]), 1) / 100
        theta0 = torch.matmul(self.r, p_inj_r.T)
        
        # ✅ 使用 self.device
        ref_ang = torch.zeros(1, theta0.shape[1]).to(self.device)
        theta = torch.cat([theta0[:68, :], ref_ang, theta0[68:, :]], 0)
        v_pred = (pred[:, self.n_bus:].transpose(0, 1)) * 0.01 + 0.9
        
        # s penalty
        theta1 = theta[self.line_loc[:, 0] - 1, :]
        theta2 = theta[self.line_loc[:, 1] - 1, :]
        V1 = (v_pred[self.line_loc[:, 0] - 1, :]).double()
        V2 = (v_pred[self.line_loc[:, 1] - 1, :]).double()
        f_p = (self.a * self.g * (V1 * V1).T) - self.a * ((V1 * V2).T) * (
            self.g * torch.cos(theta1 - theta2).T + self.b * torch.sin(theta1 - theta2).T
        )
        f_p = f_p.T
        f_q = -self.a * (V1.T) * (self.a * V1.T) * (self.b + self.c / 2) + self.a * ((V1 * V2).T) * (
            self.b * torch.cos(theta1 - theta2).T - self.g * torch.sin(theta1 - theta2).T
        )
        f_q = f_q.T
        s_pred = torch.sqrt(f_p * f_p + f_q * f_q + 1e-5) * 100
        s_penalty = torch.sigmoid(s_pred - self.s) + torch.sigmoid(-s_pred - self.s)
        s_total = torch.sum(s_penalty)

        # sji penalty
        theta1 = theta[self.line_loc[:, 1] - 1, :]
        theta2 = theta[self.line_loc[:, 0] - 1, :]
        V1 = (v_pred[self.line_loc[:, 1] - 1, :]).double()
        V2 = (v_pred[self.line_loc[:, 0] - 1, :]).double()
        fji_p = (self.a * self.g * (V1 * V1).T) - self.a * ((V1 * V2).T) * (
            self.g * torch.cos(theta1 - theta2).T + self.b * torch.sin(theta1 - theta2).T
        )
        fji_p = fji_p.T
        fji_q = -self.a * (V1.T) * (self.a * V1.T) * (self.b + self.c / 2) + self.a * ((V1 * V2).T) * (
            self.b * torch.cos(theta1 - theta2).T - self.g * torch.sin(theta1 - theta2).T
        )
        fji_q = fji_q.T
        sji_pred = torch.sqrt(fji_p * fji_p + fji_q * fji_q + 1e-5) * 100
        sji_penalty = torch.sigmoid(sji_pred - self.s) + torch.sigmoid(-sji_pred - self.s)
        sji_total = torch.sum(sji_penalty)

        return (
            self.lmda1 * mse_v
            + self.lmda2 * mse_p
            + self.lmda3 * linf_v
            + self.lmda5 * linf_p
            + self.lmda4 * s_total
            + self.lmda4 * sji_total
        )
