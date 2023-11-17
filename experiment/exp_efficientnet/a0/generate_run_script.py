import subprocess
from pathlib import Path

#set parameters
#run_file_name = "train_eval.py"
# run_file_name = "train_eval_efficient_net.py"
RUNNING_DIR = './run'
run_file_name = "../train_eval_efficient_net.py"

def create_dir_if_not_exist(dir_path):
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
create_dir_if_not_exist(RUNNING_DIR)

setting_param_dict = {
    # "source_dataset_dir": "./dataset",
    "source_dataset_dir": "/projectnb/cs640grp/materials/UBC-OCEAN_CS640",
    "local_dataset_dir": "../../../../dataset",
    "model_dir": "../model",
    #"experiment_name": "exp_1",
    # "train_image_folder": "img_size_256x256",
    # "train_image_folder": "train_images_compressed_80",
    #"image_input_size": "(256, 256)",
    "batch_size": 8,
    "num_epochs": 200,
    # "lr": 0.001,
    # --weight_decay: 0.0001
    "eval_patience": 50,
} 

setting_param_multi_combination_dict = {    
    #"experiment_name": ["efficientnet_b0", "efficientnet_b4", "efficientnet_widese_b0", "efficientnet_widese_b4"],
    "experiment_name": ["efficientnet_b0", "efficientnet_widese_b0", "efficientnet_b4", "efficientnet_widese_b4"],
    # "lr": [0.001, 0.0001],
    # "weight_decay": [0.0001, 0.00001]
}

setting_img_parallel_dict = {
    "train_image_folder": ["img_size_256x256", "img_size_512x512", "img_size_1024x1024"],
    "image_input_size": ["(256, 256)", "(512, 512)", "(1024, 1024)"],
}





# get command line arguments
command_line_args = ""
for key, value in setting_param_dict.items():
    if type(value) == str: 
        command_line_args += f" --{key} \"{value}\""
    else:
        command_line_args += f" --{key} {value}"

list_of_command_line_args = [command_line_args]
for key, value_list in setting_param_multi_combination_dict.items():
    list_command_tmp = []
    for value in value_list:
        for command_line_args in list_of_command_line_args:
            if type(value) == str:
                list_command_tmp.append(f"{command_line_args} --{key} \"{value}\"")
            else:
                list_command_tmp.append(f"{command_line_args} --{key} {value}")
    list_of_command_line_args = list_command_tmp
        
list_of_command_line_args_tmp = []
k = list(setting_img_parallel_dict.keys())[0]
len_val = len(setting_img_parallel_dict[k])
for i in range(len_val):
    command_tmp = ""
    for key, value in setting_img_parallel_dict.items():
        if type(value[i]) == str:
            command_tmp += f" --{key} \"{value[i]}\""
        else:
            command_tmp += f" --{key} {value[i]}"

    for command_line_args in list_of_command_line_args:
        list_of_command_line_args_tmp.append(f"{command_line_args}{command_tmp}")

list_of_command_line_args = list_of_command_line_args_tmp

# print("#/bin/bash -l") 

# print command
# print(f"python {run_file_name} {command_line_args}")

# print list of command
for i, command_line_args_tmp in  enumerate(list_of_command_line_args):
    print(f"python {run_file_name} {command_line_args_tmp} > log{i:02d}")
    with open(Path(RUNNING_DIR, f"run_exp{i:02}.sh"), "w") as f:
        f.write("#/bin/bash -l\n")
        f.write("#$ -pe omp 16\n")
        f.write("#$ -l mem_per_core=8G\n")
        f.write("#$ -l h_rt=24:00:00\n")
        f.write(f"python {run_file_name} {command_line_args_tmp} > log{i:02d}\n")

with open(Path(RUNNING_DIR, "run.sh"), "w") as f:
    f.write("#/bin/bash -l\n")
    for i, command_line_args_tmp in  enumerate(list_of_command_line_args):
        f.write(f"qsub run_exp{i:02}.sh\n")

#chmod 777 all file under running dir
subprocess.run(f"chmod 777 {RUNNING_DIR}/*", shell=True)


print("\n\n\n\n\n")

# # run command
# # subprocess.run(f"python {run_file_name} {command_line_args}", shell=True)

# # run list of command
# for command_line_args_tmp in list_of_command_line_args:
#     print(f"python {run_file_name} {command_line_args_tmp}")
#     subprocess.run(f"python {run_file_name} {command_line_args_tmp}", shell=True)