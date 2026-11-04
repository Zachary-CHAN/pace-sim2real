# 1. 验证环境和固定姿态
python scripts/zero_agent.py \
  --task Isaac-Pace-R1-v0 \
  --num_envs 1 \
  --video \
  --video_length 2000 \
  --video_folder videos/r1_zero_agent

# 2. 快速生成模拟数据
python scripts/pace/data_collection.py \
  --task Isaac-Pace-R1-v0 \
  --num_envs 1 \
  --duration 5 \
  --min_frequency 0.1 \
  --max_frequency 3.0

  预期生成：
  data/r1_real/chirp_data.pt

  400 Hz、2秒时，数据形状应为：
  time.shape        == (800,)
  dof_pos.shape     == (800, 12)
  des_dof_pos.shape == (800, 12)

没问题后用默认参数
python scripts/pace/data_collection.py   --task Isaac-Pace-R1-v0   --num_envs 1   --duration 20   --min_frequency 0.1   --max_frequency 10.0


# 3. 检查数据
python - <<'PY'
import torch

data = torch.load("data/r1_real/chirp_data.pt")

for key, value in data.items():
    if isinstance(value, torch.Tensor):
        print(key, value.shape, value.dtype)
    else:
        print(key, value)
PY

确认
joint_order与R1_JOINT_ORDER一致
control_dt约为0.0025
无NaN或Inf
起始目标位置约为0

# 4. 小规模运行拟合
目前没有 --max_iterations 参数时，可以先用较小种群运行，并在若干次迭代后手动停止：
python scripts/pace/fit.py \
  --task Isaac-Pace-R1-v0 \
  --num_envs 64 \
  --headless

确认能够出现：
CMA-ES optimizer initialized
Current iteration: 0
Min score: ...
Armature: ...
Viscous Friction: ...
Static/Dynamic Friction: ...
Bias: ...
Delay: ...

# 5. 正式采集和拟合
重新生成完整20秒数据：
python scripts/pace/data_collection.py \
  --task Isaac-Pace-R1-v0 \
  --num_envs 1 \
  --duration 20 \
  --min_frequency 0.1 \
  --max_frequency 10.0

然后：
python scripts/pace/fit.py \
  --task Isaac-Pace-R1-v0 \
  --num_envs 4096 \
  --headless

绘制结果：
python scripts/pace/plot_trajectory.py \
  --robot_name r1_real \
  --plot_trajectory

20260901的辨识结果(仅仿真)
Armature params: tensor([0.0017, 0.0151, 0.0078, 0.0009, 0.0137, 0.0166, 0.0032, 0.0147, 0.0195,
        0.0120, 0.0195, 0.0091], device='cuda:0')
Viscous friction params: tensor([9.7428, 1.2468, 4.6496, 1.4698, 5.2373, 7.2801, 9.7264, 4.4801, 4.7057,
        1.4017, 4.9969, 5.3679], device='cuda:0')
Static/dynamic friction params: tensor([0.2601, 0.2748, 0.2444, 0.2961, 0.2370, 0.2346, 0.2206, 0.1784, 0.0731,
        0.2232, 0.2174, 0.2985], device='cuda:0')
Encoder bias params: tensor([ 0.0118,  0.0595, -0.0819,  0.0455,  0.0465, -0.0560,  0.0018, -0.0010,
         0.0623,  0.0483,  0.0488,  0.0799], device='cuda:0')
Delay param: 4.567361354827881
