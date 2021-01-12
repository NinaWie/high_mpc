import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from high_mpc.mpc.mpc import MPC
# from torch.utils.tensorboard import SummaryWriter

from simulation.dynamics import s_dim, a_dim, dynamics

class DroneModel(nn.Module):

    def __init__(self, act_prior, horizon=20, dt=0.2):
        super(DroneModel, self).__init__()
        self.dt = dt
        self.horizon = horizon
        self.state_dim = s_dim
        U = [
            nn.Parameter(act_prior[k], requires_grad=True)
            for k in range(horizon)
        ]
        self.U = nn.ParameterList(U)

    def forward(self, current_state):
        intermediate_states = torch.zeros(self.horizon, self.state_dim)
        for i in range(self.horizon):
            # map action to range between 0 and 1
            apply_action = self.U[i]
            current_state = dynamics(current_state, apply_action, self.dt)
            intermediate_states[i] = current_state
        return intermediate_states

    def roll_model(self):
        for i in range(self.horizon-1):
            self.U[i] = self.U[i+1]

    def set_weights(self, init_actions):
        for i in range(self.horizon-1):
            self.U[i] = nn.Parameter(init_actions[i])

class NeuralControl(MPC):
    def __init__(self, init_on_mpc = False, **kwargs):
        super(NeuralControl, self).__init__(**kwargs)
        self.learning_rate = 0.5
        self.epochs = 20
        self.horizon = self._N
        self.init_on_mpc = init_on_mpc

        # prior on control signals
        self.action_prior = torch.tensor([[9.81, 0,0,0]]).repeat(self.horizon,1).double()

        # loss function weights
        # go to goal point
        self._Q_goal = torch.from_numpy(np.diag([
            100, 100, 100,  # delta_x, delta_y, delta_z
            10, 10, 10, 10, # delta_qw, delta_qx, delta_qy, delta_qz
            10, 10, 10])).double()
        # Track pendulum motion
        self._Q_pen = torch.from_numpy(np.diag([
            0, 100, 100,  # delta_x, delta_y, delta_z
            10, 10, 10, 10, # delta_qw, delta_qx, delta_qy, delta_qz
            0, 10, 10])).double()
        # cost matrix for the action
        self._Q_u = torch.from_numpy(np.diag([0.1, 0.1, 0.1, 0.1])).double() 
        # T, wx, wy, wz


        # initialize model and optimizer
        self.drone = DroneModel(self.action_prior, horizon=self.horizon, dt=self._dt)
        self.optimizer = optim.SGD(self.drone.parameters(), lr=self.learning_rate, momentum=0.9)
        

    def mpc_loss(self, states, actions, target_states, target_end_state):
        state_error = states - target_states # shape 20 x 10
        state_loss = torch.sum(torch.mm(torch.mm(state_error, self._Q_pen), state_error.t()))
        # print(state_error.size(), state_loss.size())
        target_error = states[-1] - target_end_state # shape 20 x 10
        inner_mv = torch.mv(self._Q_goal, target_error)
        target_loss = torch.sum(torch.dot(inner_mv, target_error.t()))
        # add up action losses
        action_loss = 0
        for k in range(self.horizon):
            action_error = actions[k] - self.action_prior[k]
            action_loss += torch.dot(torch.mv(self._Q_u, action_error), action_error.t())
        return torch.sum(state_loss + action_loss + target_loss)*0.00001

    def solve(self, reference_trajectory):
        """
        Optimize via gradient descent (overwrite the method from MPC class)
        """

        # use actions from mpc as prior
        if self.init_on_mpc:
            control, all_states = super().solve(reference_trajectory)
            print()
            print("control mpc", control)
            self.drone.set_weights(torch.from_numpy(all_states[:, s_dim:]))

        # transform reference to useful input
        reference_trajectory = np.array(reference_trajectory)
        input_state = torch.tensor(reference_trajectory[:s_dim])
        ref_inter_states = np.reshape(reference_trajectory[s_dim:-s_dim], (self.horizon, s_dim+3))
        ref_inter_states = torch.from_numpy(ref_inter_states[:,:s_dim])
        ref_target_state = torch.from_numpy(reference_trajectory[-s_dim:])

        # print("input")
        # print(input_state.size())
        # print(ref_inter_states.size())
        # print(ref_target_state.size())

        # print()
        
        # optimize
        losses, collect_states = [], []
        for j in range(self.epochs):
            states = self.drone(input_state)
            # actions = drone.U
            self.optimizer.zero_grad()
            # loss
            loss = self.mpc_loss(
                states, self.drone.U, ref_inter_states, ref_target_state
            )
            # mpc_loss(states, actions, target_states, target_actions)

            losses.append(loss)
            # if j == 0:
            #     plot_position(
            #         states.detach().numpy()[:, :3],
            #         os.path.join(SAVE_PATH, "states_start.png")
            #     )
            # print(loss.item())
            # backprop
            loss.backward()
            self.optimizer.step()
            # save states

        # return control signals and predicted trajectory
        control_signals = np.zeros((self.horizon, a_dim))
        for i, u_k in enumerate(self.drone.U):
            control_signals[i] = u_k.detach().numpy()
        pred_trajectory = np.hstack((states.detach().numpy(), control_signals))
        # roll control signals for next iteration
        # self.action_prior = torch.from_numpy(np.vstack((control_signals[1], control_signals[1:])))

        if self.init_on_mpc:
            print("control gd", control_signals[0])
        else:
            # if the control signals are not set by mpc, just roll them
            self.drone.roll_model()
        # set it to lower number of epoch after the first iteration
        self.epochs = 10
        return control_signals[0], pred_trajectory
