import os
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
from functools import partial
import sys
sys.path.append("/home/nina/Desktop/high_mpc")
#
from high_mpc.simulation.dynamic_gap import DynamicGap
from high_mpc.mpc.mpc import MPC
from high_mpc.neural_control import NeuralControl
from high_mpc.simulation.animation import SimVisual
#
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video', type=bool, default=False,
        help="Save the animation as a video file")
    return parser

def run_mpc(env):
    #
    env.reset()
    t, n = 0, 0
    t0 = time.time()
    while t < env.sim_T:
        t = env.sim_dt * n
        _, _, _, info = env.step()
        t_now = time.time()
        # print(t_now - t0)
	    #
        t0 = time.time()
        #
        n += 1
        update = False
        if t>= env.sim_T:
            update = True
        yield [info, t, update]

def main():
    visualize = 0
    #
    args = arg_parser().parse_args()
    #
    plan_T = 1.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.1 # Sampling time step for MPC and local planner
    so_path = "./mpc/saved/mpc_v1.so" # saved mpc model (casadi code generation)
    #
    mpc = MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    # NeuralControl(T=plan_T, dt=plan_dt, so_path=so_path)
    # MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = DynamicGap(mpc, plan_T, plan_dt)
    
    #

    if visualize:
        sim_visual = SimVisual(env)
        run_frame = partial(run_mpc, env)
        ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
                init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)
        plt.tight_layout()
        plt.show()
    else:
        for k in range(10):
            mpc = NeuralControl(T=plan_T, dt=plan_dt, so_path=so_path)
            # NeuralControl(T=plan_T, dt=plan_dt, so_path=so_path)
            # MPC(T=plan_T, dt=plan_dt, so_path=so_path)
            env = DynamicGap(mpc, plan_T, plan_dt)

            quad_pos, dists = [], []
            for (info, _, _) in run_mpc(env):
                quad = info["quad_obs"][:3]
                pend = info["pend_obs"][:3]
                together = quad.tolist() + pend.tolist()
                quad_pos.append(together)
                dists.append(np.linalg.norm(quad - pend))

            # plotting
            plt.figure(figsize=(15,8))
            labels = ["quad_x", "quad_y", "quad_z", "pend_x", "pend_y", "pend_z"]
            cols = ["-r", "-b", "-g", "--r", "--b", "--g"]
            quad_pos = np.swapaxes(np.array(quad_pos), 1,0)
            for i,data in enumerate(quad_pos):
                plt.plot(data, cols[i], label=labels[i])
            plt.plot(dists, c="black", label="distance")
            plt.ylim(-7.5, 7.5)
            plt.savefig(f"mpc_vs_gd/gd_new_{k}.png")
            np.save(f"mpc_vs_gd/gd_new_{k}.npy", quad_pos)
        

            
            
    # #
    # if args.save_video:
    #     writer = animation.writers["ffmpeg"]
    #     writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    #     ani.save("MPC_0.mp4", writer=writer)

    
if __name__ == "__main__":
    main()
