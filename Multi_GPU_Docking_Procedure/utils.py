import re
from typing import Dict, Union


def get_output_pose(output_path: str) -> Union[str, None]:
    try:
        with open(output_path, 'r') as f:
            docked_pdbqt = f.read()
        return docked_pdbqt
    except FileNotFoundError:
        return None
    
def get_output_score(output_path: str) -> Union[float, None]:
    try:
        score = float("inf")
        with open(output_path, 'r') as f:
            for line in f.readlines():
                if "REMARK VINA RESULT" in line:
                    new_score = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', line)[0]
                    score = min(score, float(new_score))
        return score
    except FileNotFoundError:
        return None

def write_conf_file(conf_str: str, config_file_path: str, args: Dict[str, str] = {}):  
    for k, v in args.items():
        conf_str += f"{str(k)} = {str(v)}\n"
    
    with open(config_file_path, 'w') as f:
        f.write(conf_str)