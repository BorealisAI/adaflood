import shutil
import glob

#log_list = glob.glob(f'logs/*_aux')
#
#for log_dir in log_list:
#    shutil.move(log_dir, f"{log_dir}10")
#    print(f"moved {log_dir} -> {log_dir}10")


#log_list = glob.glob(f'logs/reddit_*_adaflood_final/seed1/*_mdim64_*')
#for log_dir in log_list:
#    new_log_dir = f"{log_dir}_aux5"
#    shutil.move(log_dir, f"{new_log_dir}")
#    print(f"moved {log_dir} -> {new_log_dir}")


#log_list = glob.glob(f'results/cifar10_resnet18_alpha*_imb1.0_iflood_final')
#for log_dir in log_list:
#    new_log_dir = f"results/prev"
#    shutil.move(log_dir, f"{new_log_dir}")
#    print(f"moved {log_dir} -> {new_log_dir}")

log_list = glob.glob(f'logs/cifar10_resnet18_alpha*_imb1.0_iflood_final')
for log_dir in log_list:
    new_log_dir = log_dir.replace("logs/", "results/")
    shutil.copytree(log_dir, f"{new_log_dir}")
    print(f"copied {log_dir} -> {new_log_dir}")
