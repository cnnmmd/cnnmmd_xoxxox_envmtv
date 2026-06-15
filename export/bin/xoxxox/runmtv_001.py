import os
import sys
import signal
import time
import torch
import mujoco
from mujoco import viewer
from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation, TransformObservation
from metamotivo.fb_cpr.huggingface import FBcprModel

#---------------------------------------------------------------------------

flgrun = True

def endprc(numsig, frmcrr):
  global flgrun
  flgrun = False

for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
  signal.signal(s, endprc)

device = "cpu"

#---------------------------------------------------------------------------

env, _ = make_humenv(
  num_envs=1,
  wrappers=[
    FlattenObservation,
    lambda env: TransformObservation(
      env,
      lambda obs: torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=device),
      env.observation_space,
    ),
  ],
  state_init="Default",
)
e = env.unwrapped
mmodel = e.model
mjdata  = e.data

fmodel = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(device)
z = fmodel.sample_z(1)

with viewer.launch_passive(mmodel, mjdata) as v:
  v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  v.cam.trackbodyid = -1
  v.cam.distance = 6.0
  v.cam.azimuth = 120.0
  v.cam.elevation = -30.0
  v.sync()

  observation, info = env.reset()

  while flgrun and v.is_running() and os.getppid() != 1:
    # 調整：描画
    timbgn = time.perf_counter()
    frmbgn = float(mjdata.time)
    #

    action = fmodel.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())
  
    if terminated or truncated:
      observation, info = env.reset()
  
    v.sync()

    # 調整：描画
    frmdif = float(mjdata.time) - frmbgn
    timdif = time.perf_counter() - timbgn
    if timdif < frmdif:
      time.sleep(frmdif - timdif)
