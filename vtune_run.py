import os
import subprocess
import re
import yaml

new_cpu_infer_values = [12,24,36,46]
expert_nums = [1, 2]

def modify_config_yaml(file_path, new_value):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    config['ext']['cpu_infer'] = new_value
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

def modify_local_chat_py(file_path, new_value):
    with open(file_path, 'r') as file:
        content = file.read()
    content = re.sub(r'torch.randint\(1, 63, torch.Size\(\[\d+\]\)\)', f'torch.randint(1, 63, torch.Size([{new_value}]))', content)
    with open(file_path, 'w') as file:
        file.write(content)

def sudo_set_value(file_path, value):
    subprocess.run(['sudo', 'sh', '-c', f'echo {value} > {file_path}'], check=True)


def source_activate():
    subprocess.run(['source', '/data/yanfansun/.conda/envs/ktransformers/lib/python3.11/venv/scripts/common/activate'], check=True)

for cpu_infer in new_cpu_infer_values:
    for expert_num in expert_nums:
        modify_config_yaml('/data/yanfansun/ktrans/ktransformers/ktransformers/configs/config.yaml', cpu_infer)
        modify_local_chat_py('/data/yanfansun/ktrans/ktransformers/ktransformers/local_chat.py', expert_num)
        #source_activate()
        sudo_set_value('/proc/sys/kernel/perf_event_paranoid', '0')
        sudo_set_value('/proc/sys/kernel/kptr_restrict', '0')

        result_dir = f'/data/yanfansun/ktrans/ktransformers/vtune_result/all_{cpu_infer}_{expert_num}'

        vtune_command = [
            'vtune', '-collect', 'uarch-exploration',
            '-resume-after=30', f'-result-dir={result_dir}',
            'python3', '/data/yanfansun/ktrans/ktransformers/ktransformers/local_chat.py'
        ]
        # vtune -collect hotspots xxx
        subprocess.run(vtune_command)


        sudo_set_value('/proc/sys/kernel/perf_event_paranoid', '2')
        sudo_set_value('/proc/sys/kernel/kptr_restrict', '1')