import subprocess

#set parameters
run_file_name = "resize_dataset_image.py"
# run_file_name = "train_eval_efficient_net.py"

setting_param_dict = {
    # "source_dataset_dir": "/projectnb/cs640grp/materials/UBC-OCEAN_CS640",
    "source_dataset_dir": "./dataset/",
    "target_dataset_dir": "./dataset_official",
    # "model_dir": "./model",
    # "source_train_image_folder": "train_images_compressed_80",
    "source_train_image_folder": "train_thumbnails",
    "source_test_image_folder": "test_images_compressed_80",
} 

setting_param_multi_combination_dict = {
    # "image_downsampled_size": ["(256, 256)", "(512, 512)", "(1024, 1024)", "(2048, 2048)"]
    "image_downsampled_size": ["(256, 256)", "(512, 512)"]
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
# print(f"python {run_file_name} {command_line_args}")

# # print list of command
for command_line_args_tmp in list_of_command_line_args:
    print(f"python {run_file_name} {command_line_args_tmp}")

# run command
# subprocess.run(f"python {run_file_name} {command_line_args}", shell=True)

# # run list of command
for command_line_args_tmp in list_of_command_line_args:
    subprocess.run(f"python {run_file_name} {command_line_args_tmp}", shell=True)