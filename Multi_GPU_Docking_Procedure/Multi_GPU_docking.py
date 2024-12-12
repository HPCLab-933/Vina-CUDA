import os
import subprocess
import re
import time
import shutil
import hashlib

from typing import Dict, Iterable, List, Union
from rdkit import Chem
#import ligand_preparators
from utils import get_output_pose, get_output_score, write_conf_file

class OutOfGPUMemoryError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"{self.message}"
    
class OpenCLBinaryError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"{self.message}"

def delete_all():
    # DEBUGGING PURPOSES
    if os.path.exists('./configs/'):
        shutil.rmtree('./configs') 
    if os.path.exists('./ligands/'):
        shutil.rmtree('./ligands/')
    if os.path.exists('./ligands_tmp/'):
        shutil.rmtree('./ligands_tmp/')
    if os.path.exists('./outputs_tmp/'):
        shutil.rmtree('./outputs_tmp/')
    if os.path.exists('./logs/'):
        shutil.rmtree('./logs/')
    if os.path.exists('./outputs/'):
        shutil.rmtree('./outputs/')

def make_dir(rel_path, *args, **kwargs):
    os.makedirs(os.path.abspath(rel_path), *args, **kwargs)

def sanitize_smi_name_for_file(smi: str):
    """
        Sanitization for file names. Replacement values cannot be part of valid SMILES.
    """
    return hashlib.sha224(smi.encode()).hexdigest()

def move_files_from_dir(source_dir_path: str, dest_dir_path: str):
    files = os.listdir(source_dir_path)
    for file in files:
        source_file = os.path.join(source_dir_path, file)
        destination_file = os.path.join(dest_dir_path, file)
        shutil.move(source_file, destination_file)

def delete_dir_contents(dir_path: str):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def split_list(lst, n):
    """Split a list into n equal parts, with any remainder added to the last split."""
    if n <= 0:
        raise ValueError("Number of splits (n) must be a positive integer.")
    
    quotient, remainder = divmod(len(lst), n)
    splits = [lst[i * quotient + min(i, remainder):(i + 1) * quotient + min(i + 1, remainder)] for i in range(n)]
    return splits


class TimedProfiler:
    def __init__(self) -> None:
        self._count = 0
        self._total = 0
        self._average = 0
    
    def _add_value(self, value):
        self._total += value
        self._count += 1
        self._average = self._total / self._count
    
    def time_it(self, fn, *args, **kwargs):
        start_time = time.time()
        res = fn(*args, **kwargs)
        end_time = time.time()
        self._add_value(end_time - start_time)
        return res
    
    def get_average(self):
        return self._average


class VinaDocking:
    def __init__(self, 
                 vina_cmd: str,
                 receptor_pdbqt_file: str,
                 center_pos: List[float],
                 size: List[float],
                 ligand_dir_path: str = 'ligands/',
                 output_dir_path: str = 'outputs/',
                 log_dir_path: str = 'logs/',
                 config_dir_path: str = 'configs/',
                 tmp_ligand_dir_path: str = 'ligands_tmp/',
                 tmp_output_dir_path: str = 'outputs_tmp/',
                 tmp_config_file_path: str = 'config_tmp_conf',
                 keep_ligand_file: bool = True,
                 keep_output_file: bool = False,
                 keep_log_file: bool = True,
                 keep_config_file: bool = True,
                 get_pose_str: bool = False,
                 timeout_duration: int = None,
                 additional_vina_args: Dict[str, str] = {},
                 #ligand_preparation_fn: callable = ligand_preparators.MeekoLigandPreparator(False),
                 vina_cwd: str = None,
                 gpu_ids: Union[int, List[int]] = 0,
                 print_msgs: bool = True,
                 print_vina_output: bool = True,
                 debug: bool = False) -> None:
        """
            Parameters:
            - vina_cmd: Command line prefix to execute vina command (e.g. "/path/to/qvina2.1")
            - receptor_pdbqt_file: Cleaned receptor PDBQT file to use for docking
            - center_pos: 3-dim list containing (x,y,z) coordinates of grid box
            - size: 3-dim list containing sizing information of grid box in (x,y,z) directions
            - ligand_dir_path: Path to save ligand preparation files
            - output_dir_path: Path to save docking output files
            - log_dir_path: Path to save log files
            - config_dir_path: Path to save config files
            - tmp_ligand_dir_path: Path to save temporary ligand files (for batched docking)
            - tmp_output_dir_path: Path to save temporary output files (for batched docking)
            - tmp_config_file_path: Path to save temporary config file (for batched docking)
            - keep_ligand_file: Save ligand file (True) or not (False)
            - keep_output_file: Save output file (True) or not (False)
            - keep_log_file: Save log file (True) or not (False)
            - keep_config_file: Save config file (True) or not (False)
            - get_pose_str: Return output pose as string (True) or not (False)
            - timeout_duration: Timeout in seconds before new process automatically stops
            - additional_vina_args: Dictionary of additional Vina command arguments (e.g. {"cpu": "5"})
            - ligand_preparation_fn: Function/Class callable to prepare molecule for docking. Should take the \
                argument format (smiles strings, ligand paths)
            - vina_cwd: Change current working directory of Vina shell (sometimes needed for GPU versions \
                and incorrect openCL pathing)
            - gpu_ids: GPU ids to use for multi-GPU docking (0 is default for single-GPU nodes). If None, \
                use all GPUs.
            - print_msgs: Show Python print messages in console (True) or not (False)
            - print_vina_output: Show Vina docking output in console (True) or not (False)
            - debug: Profiling the Vina docking process and ligand preparation.
        """

        # Check if certain docking aspects are supported, such as logging and batched docking
        # (has vina command arguments e.g. '--log')
        try:
            with subprocess.Popen(f"{vina_cmd} --help",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    cwd=vina_cwd) as proc:
                if proc.wait(timeout=timeout_duration) == 0:
                    result = proc.stdout.read()
                    if result is not None:
                        result = str(result)
                        self.batch_docking_support = False
                        self.logging_support = False

                        if "--log" in result:
                            self.logging_support = True
                        if "--ligand_directory" and "--output_directory" in result:
                            self.batch_docking_support = True
                else:
                    raise Exception(f"Vina command '{vina_cmd}' returned unsuccessfully: {proc.stderr.read()}")
        except subprocess.TimeoutExpired:
            proc.kill()
            self.logging_support = False
            self.batch_docking_support = False
        
        if not os.path.isfile(receptor_pdbqt_file):
            raise Exception(rf'Receptor file: {receptor_pdbqt_file} not found')
        
        if not os.path.isdir(ligand_dir_path):
            raise Exception(rf'ligand file: {ligand_dir_path} not found')

        if len(center_pos) != 3:
            raise Exception(f"center_pos must contain 3 values: {center_pos} was provided")

        if len(size) != 3:
            raise Exception(f"size must contain 3 values: {size} was provided")
        
        if print_msgs and not self.logging_support:
            print("Log files are not supported with this Vina command")

        if print_msgs and self.batch_docking_support:
            print("Batched docking is enabled; log and config files will not be saved")
        elif print_msgs and not self.batch_docking_support:
            print("Batched docking is disabled; not supported with selected Vina version")

        # Getting all available GPU ids
        with subprocess.Popen(f"nvidia-smi --list-gpus",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True) as proc:
            if proc.wait(timeout=timeout_duration) == 0:
                out = proc.stdout.read().decode('ascii')
                pattern = r"GPU (\d+):"
                available_gpu_ids = [int(x) for x in re.findall(pattern, out)]
            else:
                raise Exception(f"Command 'nvidia-smi --list-gpus' returned unsuccessfully: {proc.stderr.read()}")
        
        # Checking for incorrect GPU id input
        if gpu_ids is None:
            gpu_ids = available_gpu_ids
        elif type(gpu_ids) is int:
            if gpu_ids in available_gpu_ids:
                gpu_ids = [self.gpu_ids]
            else:
                raise Exception(f"Unknown GPU id: {gpu_ids}")
        else:
            unknown_gpu_ids = []
            for gpu_id in gpu_ids:
                if gpu_id not in available_gpu_ids:
                    unknown_gpu_ids.append(gpu_id)
            if len(unknown_gpu_ids) > 0:
                raise Exception(f"Unknown GPU id(s): {unknown_gpu_ids}")
            
        if self.batch_docking_support:
            self.logging_support = False

        self.vina_cmd = vina_cmd
        self.receptor_pdbqt_file = os.path.abspath(receptor_pdbqt_file)
        self.center_pos = center_pos
        self.size = size
        self.ligand_dir_path = ligand_dir_path
        self.output_dir_path = output_dir_path
        self.log_dir_path = log_dir_path
        self.config_dir_path = config_dir_path
        self.tmp_ligand_dir_path = tmp_ligand_dir_path
        self.tmp_output_dir_path = tmp_output_dir_path
        self.tmp_config_file_path = tmp_config_file_path
        self.keep_ligand_file = keep_ligand_file
        self.keep_output_file = keep_output_file
        self.keep_log_file = keep_log_file
        self.keep_config_file = keep_config_file
        self.get_pose_str = get_pose_str
        self.timeout_duration = timeout_duration
        #self.ligand_preparation_fn = ligand_preparation_fn  # should follow format (smiles string, ligand_path)
        self.vina_cwd = vina_cwd
        self.gpu_ids = gpu_ids
        self.print_msgs = print_msgs
        self.print_vina_output = print_vina_output
        self.debug = debug

        if debug:
            self.preparation_profiler = TimedProfiler()
            self.docking_profiler = TimedProfiler()
        
        self.conf_str = f'receptor = {self.receptor_pdbqt_file}\n' + \
                        f'center_x = {self.center_pos[0]}\n' + \
                        f'center_y = {self.center_pos[1]}\n' + \
                        f'center_z = {self.center_pos[2]}\n' + \
                        f'size_x = {self.size[0]}\n' + \
                        f'size_y = {self.size[1]}\n' + \
                        f'size_z = {self.size[2]}\n'
        
        for k, v in additional_vina_args.items():
            self.conf_str += f"{str(k)} = {str(v)}\n"


    def __call__(self,path):
        if os.path.exists(path):
            if os.path.isfile(path):   
                return self._docking(path)         
            elif os.path.isdir(path):
                return self._batched_docking(path)
            else:
                print(f"{path} is not effetive a file or the folder")
        elif isinstance(path, str) and is_smiles(path):
            if type(smi) is str:
                smi = [smi]

            for i in range(len(smi)):
                mol = Chem.MolFromSmiles(smi[i])
                if mol is not None:
                    smi[i] = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            
            self._make_dirs()

            if self.batch_docking_support and len(smi) > 1:
                return self._batched_docking(smi)
            else:
                return self._docking(smi)
        else:
             print(f"{path} is not correct path, please check again !!!!!!!!!!!!")


    def _docking(self, path):
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        ligand_name = os.path.splitext(filename)[0]

        ligand_path = os.path.abspath(f"{directory}/{ligand_name}.pdbqt")
        output_path = os.path.abspath(f"{directory}/{ligand_name}_out.pdbqt")
        config_path = os.path.abspath(f"{directory}/{ligand_name}_conf.txt")
        log_path = os.path.abspath(f"{directory}/{ligand_name}_log.txt")   
    
       
        write_conf_file(self.conf_str, config_path, args={"ligand": ligand_path, "out": output_path})

        if self.logging_support and self.keep_log_file:
            log_path = log_path
        else:
            log_path = None

        print("********** [All parses have been configured and run docking] **********")

        if self.debug:
            self.docking_profiler.time_it(self._run_vina, [config_path], [log_path])
        else:
            self._run_vina([config_path], [log_path])
        
        # Gather binding scores
        binding_scores = []
        
        binding_scores.append(get_output_score(output_path))

        # Gather binding poses
        if self.get_pose_str:
            binding_poses = []
            binding_poses.append(get_output_pose(output_path))
            
            binding_scores = list(zip(binding_scores, binding_poses))

       
        if not self.keep_config_file and os.path.exists(config_path):
            os.remove(config_path)
        if not self.keep_ligand_file and os.path.exists(ligand_path):
            os.remove(ligand_path)
        if not self.keep_output_file and os.path.exists(output_path):
            os.remove(output_path)

        print(f"\nThe best score of {ligand_name} is: {binding_scores}")
        #return binding_scores

    def _batched_docking(self, paths: Iterable[str]) -> List[Union[float, None]]:
        # TODO: GPU versions can be made faster by better pipelining (prepare x number of molecules on CPU while docking at same time on GPU)

        self._make_tmp_files()

        ligand_names = []
        #directory = os.path.dirname(paths)
        for path in os.listdir(paths):
            if os.path.isfile(os.path.join(paths, path)):
                filename = os.path.basename(path)
                src_file = os.path.join(os.path.abspath(f"{self.ligand_dir_path}"), filename)
                dest_file = os.path.join(os.path.abspath(f"{self.tmp_ligand_dir_path}"), filename)
                shutil.copy2(src_file, dest_file)
                ligand_names.append(os.path.splitext(filename)[0])

        tmp_ligand_path_fn = lambda temp: os.path.abspath(f"{self.tmp_ligand_dir_path}{temp}.pdbqt")
        ligand_path_fn = lambda temp: os.path.abspath(f"{self.ligand_dir_path}/{temp}.pdbqt")
        output_path_fn = lambda temp: os.path.abspath(f"{self.output_dir_path}{temp}_out.pdbqt")
        tmp_output_path_fn = lambda temp: os.path.abspath(f"{self.tmp_output_dir_path}{temp}_out.pdbqt")
        
        # not the fastest implementation, but safe if multiple experiments running at same time (with different tmp file paths)
        overlap_idxs = [i for i in range(len(ligand_names)) if os.path.exists(output_path_fn(ligand_names[i]))]
        non_overlap_ligand_paths = [ligand_path_fn(ligand_names[i]) for i in range(len(ligand_names)) if i not in overlap_idxs]
        non_overlap_tmp_ligand_paths = [tmp_ligand_path_fn(ligand_names[i]) for i in range(len(ligand_names)) if i not in overlap_idxs]
        non_overlap_names = [ligand_names[i] for i in range(len(ligand_names)) if i not in overlap_idxs]

    

        if self.keep_output_file:
            output_paths = [output_path_fn(name) for name in ligand_names]
        else:
            # If already calculated, will read from output_dir, but if not will save to tmp_output_dir, read output, and then delete the tmp file
            output_paths = [output_path_fn(ligand_names[i]) if i in overlap_idxs else tmp_output_path_fn(ligand_names[i]) for i in range(len(ligand_names))]
    
        if self.keep_output_file:
            output_dir = self.output_dir_path
        else:
            output_dir = self.tmp_output_dir_path


        # Prepare ligands that don't have an existing output file (they aren't overlapping)
        #self._prepare_ligands(non_overlap_smis, non_overlap_ligand_paths, non_overlap_tmp_ligand_paths)

        # GPU docking: move ligands to the gpu_id directories and write config files for docking
        # For multi-GPU, will use multiple temporary directories
        print(f"the GPU number is : {len(self.gpu_ids)}")
        split_tmp_ligand_paths = split_list(non_overlap_tmp_ligand_paths, len(self.gpu_ids))


        tmp_config_file_paths = []
        for i in range(len(self.gpu_ids)):
            gpu_id = self.gpu_ids[i]
            tmp_config_file_path = f"{self.tmp_config_file_path}_{gpu_id}"
            outputs_config_file_path = f"{self.tmp_output_dir_path}{gpu_id}"
            tmp_config_file_paths.append(tmp_config_file_path)
            abs_tmp_ligand_dir_path = os.path.abspath(os.path.join(self.tmp_ligand_dir_path, str(gpu_id)))
            
            print(f"\n{abs_tmp_ligand_dir_path}")

            write_conf_file(self.conf_str, tmp_config_file_path,
                            {"ligand_directory": abs_tmp_ligand_dir_path,
                             "output_directory": outputs_config_file_path})
            for tmp_ligand_file in split_tmp_ligand_paths[i]:
                try:
                    shutil.copy(tmp_ligand_file, os.path.abspath(abs_tmp_ligand_dir_path))
                except FileNotFoundError:
                    if self.print_msgs:
                        print(f"Ligand file not found: {tmp_ligand_file}")

        # Perform docking procedure(s)
        vina_cmd_prefixes = [f"CUDA_VISIBLE_DEVICES={gpu_id} " for gpu_id in self.gpu_ids]
        if self.debug:
            self.docking_profiler.time_it(self._run_vina, tmp_config_file_paths, vina_cmd_prefixes=vina_cmd_prefixes, blocking=False)
        else:
            self._run_vina(tmp_config_file_paths, vina_cmd_prefixes=vina_cmd_prefixes, blocking=False)

        # Move files from temporary to proper directory (or delete if redoing calculation)
        if self.keep_ligand_file:
            for gpu_id in self.gpu_ids:
                move_files_from_dir(f"{self.tmp_ligand_dir_path}{gpu_id}/", self.ligand_dir_path)

        if self.keep_output_file:
            for gpu_id in self.gpu_ids:
                move_files_from_dir(f"{self.tmp_output_dir_path}{gpu_id}", self.output_dir_path)

        # Gather binding scores
        binding_scores = []
        for i in range(len(ligand_names)):
            binding_scores.append(get_output_score(output_paths[i]))

        # Gather binding poses
        if self.get_pose_str:
            binding_poses = []
            for i in range(len(ligand_names)):
                binding_poses.append(get_output_pose(output_paths[i]))
            
            binding_scores = list(zip(binding_scores, binding_poses))

        self._delete_tmp_files(tmp_config_file_paths)

        print(f"The final docking results are in: {os.path.abspath(self.output_dir_path)}")

        return binding_scores
    
    def _prepare_ligands(self, smis: List[str],
                               ligand_paths: List[str],
                               tmp_ligand_paths: List[str] = None) -> List[bool]:
        # Copy file from ligand_paths to tmp_ligand_paths (batched docking)
        if tmp_ligand_paths is not None:
            for i in range(len(ligand_paths)):
                if os.path.isfile(ligand_paths[i]):
                    shutil.copy(ligand_paths[i], tmp_ligand_paths[i])
                    if self.print_msgs:
                        print(f'Ligand file: {ligand_paths[i]!r} already exists, copying to {tmp_ligand_paths[i]!r}')

        # Perform ligand preparation and save to proper path (tmp/non-tmp ligand dir)
        if tmp_ligand_paths is not None:
            save_ligand_path = tmp_ligand_paths
        else:
            save_ligand_path = ligand_paths
        
        if self.debug:
            return self.preparation_profiler.time_it(self.ligand_preparation_fn, smis, save_ligand_path)
        else:
            return self.ligand_preparation_fn(smis, save_ligand_path)
    
    def _run_vina(self, config_paths: List[List[str]], log_paths: List[List[str]] = None, vina_cmd_prefixes: List[str] = None, blocking: bool = True):
        """
            Runs Vina docking in separate shell process(es).
        """
        if log_paths is not None:
            assert len(config_paths) == len(log_paths)
        if vina_cmd_prefixes is not None:
            assert len(config_paths) == len(vina_cmd_prefixes)
        procs = []

        for i in range(len(config_paths)):
            if vina_cmd_prefixes is not None and vina_cmd_prefixes[i] is not None:
                cmd_str = vina_cmd_prefixes[i]
            else:
                cmd_str = ""

            cmd_str += f"{self.vina_cmd} --config {config_paths[i]}"

            if self.logging_support and log_paths[i] is not None:
                cmd_str += f" --log {log_paths[i]}"
            if not self.print_vina_output:
                cmd_str += " > /dev/null 2>&1"

            proc = subprocess.Popen(cmd_str, shell=True, start_new_session=False, cwd=self.vina_cwd)
            if blocking:
                try:
                    proc.wait(timeout=self.timeout_duration)
                except subprocess.TimeoutExpired:
                    proc.kill()
            else:
                procs.append(proc)
        
        if not blocking:
            for proc in procs:
                try:
                    proc.wait(timeout=self.timeout_duration)
                except subprocess.TimeoutExpired:
                    proc.kill()
    
    def _make_dirs(self):
        make_dir(self.output_dir_path, exist_ok=True)
        make_dir(self.ligand_dir_path, exist_ok=True)
        if self.logging_support and self.keep_log_file:
            make_dir(self.log_dir_path, exist_ok=True)
        make_dir(self.config_dir_path, exist_ok=True)
    
    def _delete_tmp_files(self, tmp_config_file_paths: List[str] = None):
        if tmp_config_file_paths is not None:
            for tmp_config_file_path in tmp_config_file_paths:
                if os.path.exists(tmp_config_file_path):
                    os.remove(tmp_config_file_path)

        if os.path.exists(self.tmp_config_file_path):
            os.remove(self.tmp_config_file_path)
        
        if os.path.exists(self.tmp_ligand_dir_path):
            shutil.rmtree(self.tmp_ligand_dir_path)
        
        if os.path.exists(self.tmp_output_dir_path):
            shutil.rmtree(self.tmp_output_dir_path)
    
    def _make_tmp_files(self):
        # make temporary ligand and output directories
        make_dir(self.tmp_ligand_dir_path, exist_ok=True)
        make_dir(self.tmp_output_dir_path, exist_ok=True)
        make_dir(self.output_dir_path, exist_ok=True)

        # Multi-GPU: multiple ligand directories
        for gpu_id in self.gpu_ids:
            make_dir(f"{self.tmp_ligand_dir_path}/{gpu_id}/", exist_ok=True)
        
    def delete_files(self):
        self._delete_tmp_files()
        
        if os.path.exists(self.ligand_dir_path):
            shutil.rmtree(self.ligand_dir_path)
        
        if os.path.exists(self.log_dir_path):
            shutil.rmtree(self.log_dir_path)
        
        if os.path.exists(self.config_dir_path):
            shutil.rmtree(self.config_dir_path)
        
        if os.path.exists(self.output_dir_path):
            shutil.rmtree(self.output_dir_path)

#You need to enter the absolute path of the docking program, the small molecule library, and the protein.
if __name__ == "__main__":
    docking_module_gpu = VinaDocking("./Autodock-Vina-CUDA-master/Vina-GPU-2-1-CUDA",   #the Path to the executable docking program
                                 "./Autodock-Vina-CUDA-master/input_file_example/PPARG/5Y2T_receptor.pdbqt",  # The path of receptor proteins
                                 [28.000, 3.000, 28.010],
                                 [25.0, 25.00, 25.00],
                                 "./Autodock-Vina-CUDA-master/input_file_example/PPARG/actives_pdbqt", #The path of ligand compounds
                                 keep_ligand_file=True,
                                 keep_config_file=True,
                                 keep_log_file=True,
                                 keep_output_file=True,
                                 timeout_duration=None,
                                 gpu_ids=None, # None means detecting all available GPU IDs and using them
                                 additional_vina_args={"thread": "8192"})
    try:                         
        print(docking_module_gpu("./Autodock-Vina-CUDA-master/input_file_example/PPARG/actives_pdbqt")) #The path of ligand compounds
    except OutOfGPUMemoryError:
        print(docking_module_gpu("./Autodock-Vina-CUDA-master/input_file_example/PPARG/actives_pdbqt")) #The path of ligand compounds