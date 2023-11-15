import os
import pickle
import numpy as np

aux_num = 2
alpha = 0.2
loss_dir = f"/home/whbae/meta-tpp-lightning/outputs/cifar10_resnet18_alpha{alpha}_imb1.0_cls_aux{aux_num}/seed1/lr0.1_wd0.0_mdim64"

print(f"loss dir: {loss_dir}")
eval_loss_path = os.path.join(loss_dir, f"aux{aux_num}_eval_losses.pkl")
train_loss_path = os.path.join(loss_dir, f"aux{aux_num}_train_losses.pkl")

with open(eval_loss_path, "rb") as f:
    eval_loss_dict = pickle.load(f)

with open(train_loss_path, "rb") as f:
    train_loss_dict = pickle.load(f)

eval_losses = []
train_losses = []
for idx in eval_loss_dict.keys():
    eval_loss = eval_loss_dict[idx]
    train_loss = train_loss_dict[idx]

    eval_losses.append(eval_loss)
    train_losses.append(train_loss)

eval_losses = np.stack(eval_losses)
train_losses = np.stack(train_losses)

diff = eval_losses - train_losses
pos_diff = diff[diff > 0]
neg_diff = diff[diff < 0]
neg_diff.sort()
neg_train_losses = train_losses[diff < 0]
neg_train_losses_idx = np.argsort(neg_train_losses)
neg_eval_losses = eval_losses[diff < 0][neg_train_losses_idx]

print(np.mean(pos_diff), pos_diff.max())
print(np.mean(neg_diff), neg_diff.min())
#print(neg_eval_losses[-100:])

num_higher_train_losses = np.sum(train_losses > eval_losses)

print(f"Num of higher train losses than eval losses: {num_higher_train_losses}/{eval_losses.shape[0]}")

lb = np.where(eval_losses < train_losses, eval_losses, train_losses)
ub = eval_losses

print(np.sum(ub < lb))



