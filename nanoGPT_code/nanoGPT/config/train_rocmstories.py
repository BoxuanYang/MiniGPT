# config/train_rocmstories.py

out_dir = 'out-rocmstories'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'rocmstories-char'
wandb_run_name = 'mini-gpt-rocmstories'

dataset = 'rocmstories'

gradient_accumulation_steps = 1
batch_size = 64
block_size = 128  # shorter context

# baby GPT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 1e-3
max_iters = 6000
lr_decay_iters = 6000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# debug-friendly
compile = False