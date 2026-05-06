import sys
import signal
import time
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import mujoco
from mujoco import viewer
import humenv
from humenv import make_humenv
from gymnasium.wrappers import FlattenObservation, TransformObservation
from metamotivo.fb_cpr.huggingface import FBcprModel

#---------------------------------------------------------------------------

flgrun = True

def endprc():
  global flgrun
  flgrun = False

for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
  signal.signal(s, endprc)

device = "cpu"

humtgt = sys.argv[1] # "Pelvis"
sphrad = float(sys.argv[2]) # 0.1
sphmas = float(sys.argv[3]) # 20.0
sphvlo = float(sys.argv[4]) # 30.0
sphtrw = float(sys.argv[5]) # 2.0

#---------------------------------------------------------------------------

# 環境に球体を追加
def bldxml(rad: float, mas: float) -> str:
  pthorg = Path(humenv.__file__).resolve().parent / "assets" / "robot.xml" # 環境のＸＭＬ
  objxml = ET.parse(pthorg)
  xmltop = objxml.getroot()
  xmlbdy = xmltop.find("worldbody")
  xmladd = ET.fromstring(f"""
    <body name="throw_ball" pos="0.0 0.0 100.0">
      <freejoint/>
      <geom name="throw_ball_geom"
      contype="65535"
      conaffinity="65535"
      type="sphere"
      size="{rad}"
      mass="{mas}"
      rgba="0.5 0.5 0.5 1"/>
    </body>
  """)
  xmlbdy.append(xmladd)
  return ET.tostring(xmltop, encoding="unicode")

# ＸＭＬをパスに格納
def conpth(xml: str) -> str:
  tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
  tmp.write(xml)
  tmp.flush()
  tmp.close()
  pth = tmp.name
  return pth

# カメラの位置を取得
def getcam(cam):
  azi = np.deg2rad(cam.azimuth)
  elv = np.deg2rad(cam.elevation)
  fwd = np.array([
    np.cos(elv) * np.cos(azi),
    np.cos(elv) * np.sin(azi),
    np.sin(elv),
  ])
  return np.array(cam.lookat) - cam.distance * fwd

# 人体の位置を取得
def gethmn(hum):
  bid = mujoco.mj_name2id(mmodel, mujoco.mjtObj.mjOBJ_BODY, hum)
  pos = mjdata.xpos[bid].copy()
  return pos

# 球体のふるまいを取得
def getsph(sph):
  bid = mujoco.mj_name2id(mmodel, mujoco.mjtObj.mjOBJ_BODY, sph)
  jid = mmodel.body_jntadr[bid]
  qad = mmodel.jnt_qposadr[jid] # [x y z qw qx qy qz]
  dad = mmodel.jnt_dofadr[jid] # [vx vy vz wx wy wz]
  return (qad, dad)

# 球体を投げる
def trwsph(cam, hum, sph, vlo):
  poscam = getcam(cam)
  poshmn = gethmn(hum)
  d = poshmn - poscam
  n = np.linalg.norm(d)
  if n < 1e-9:
    return
  u = d / n
  qad, dad = getsph(sph)
  mjdata.qpos[qad:qad+3] = poscam
  mjdata.qpos[qad+3:qad+7] = [1.0, 0.0, 0.0, 0.0]
  mjdata.qvel[dad:dad+3] = vlo * u
  mjdata.qvel[dad+3:dad+6] = 0.0  # 回転なし
  mujoco.mj_forward(mmodel, mjdata)

#---------------------------------------------------------------------------

strxml = bldxml(sphrad, sphmas)
pthxml = conpth(strxml)

env, _ = make_humenv(
  num_envs=1,
  xml=pthxml, # 変更した環境を読み込み
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
mjdata = e.data

fmodel = FBcprModel.from_pretrained("facebook/metamotivo-M-1").to(device)
z = fmodel.sample_z(1)

with viewer.launch_passive(mmodel, mjdata) as v:
  v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  v.cam.trackbodyid = -1
  v.cam.lookat[:] = np.array([0.0, 0.0, 1.0])
  v.cam.distance = 6.0
  v.cam.azimuth = 120.0
  v.cam.elevation = -30.0
  v.sync()

  observation, info = env.reset()
  frmtrw = float(mjdata.time)

  while v.is_running() and flgrun and os.getppid() != 1:
    # 調整：描画
    timbgn = time.perf_counter()
    frmbgn = float(mjdata.time)
    #
    action = fmodel.act(observation, z, mean=True)
    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy().ravel())

    if float(mjdata.time) - frmtrw >= sphtrw:
      trwsph(v.cam, humtgt, "throw_ball", sphvlo)
      frmtrw = float(mjdata.time)

    if terminated or truncated:
      observation, info = env.reset()
      frmtrw = float(mjdata.time)

    v.sync()

    # 調整：描画
    frmdif = float(mjdata.time) - frmbgn
    timdif = time.perf_counter() - timbgn
    if timdif < frmdif:
      time.sleep(frmdif - timdif)
