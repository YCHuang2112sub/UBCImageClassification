import subprocess

#set parameters
run_file_name = "train_eval.py"
# run_file_name = "train_eval_efficient_net.py"

setting_param_dict = {
    # "source_dataset_dir": "./dataset",
    "source_dataset_dir": "/projectnb/cs640grp/materials/UBC-OCEAN_CS640",
    "local_dataset_dir": "./dataset",
    "model_dir": "./model",
    "experiment_name": "exp_1",
    "train_image_folder": "img_size_256x256",
    # "train_image_folder": "train_images_compressed_80",
    "image_input_size": "(256, 256)",
    "batch_size": 32,
    "num_epochs": 20,
    # "lr": 0.001,
    # --weight_decay: 0.0001
} 

setting_param_multi_combination_dict = {
    "lr": [0.001, 0.0001],
    "weight_decay": [0.0001, 0.00001]
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
        

# print command
print(f"python {run_file_name} {command_line_args}")

# # print list of command
# for command_line_args_tmp in list_of_command_line_args:
#     print(f"python {run_file_name} {command_line_args_tmp}")

# run command
subprocess.run(f"python {run_file_name} {command_line_args}", shell=True)

# # run list of command
# for command_line_args_tmp in list_of_command_line_args:
#     subprocess.run(f"python {run_file_name} {command_line_args_tmp}", shell=True)