# config/finetune_task2.py

# 1. 来源：从 Task 1 的结果恢复
init_from = 'resume'
# 这里填写你 Task 1 存放 ckpt.pt 的原始目录
out_dir = 'out-rocmstories' 


# 2. 目标：将微调后的模型存放在新地方，避免覆盖 Task 1
# 修改这行：
out_dir_finetune = 'out-rocmstories-task2'
is_finetune = True

# 3. 数据：指定 Task 2 专属文件
dataset = 'task2'


# 4. 训练逻辑：
# 我们把总步数设为 Task 1 已有步数 + 想要微调的步数
# 假设 Task 1 停在 14250，我们想再跑 5000 步
max_iters = 20000 


always_save_checkpoint = True
learning_rate = 1e-4
decay_lr = False
batch_size = 32
block_size = 128

val_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'finetune-2'
wandb_run_name = 'ft-' + str(time.time())

