def gen_slurm_scripts(
        script_name: str,
        job_name: str,
        time: str,
        partition: str,
        n_nodes: int,
        n_cpus: int,
        mem: str,
        n_gpus: int,
        commands: str,
        exclude_nodes: str=None,
        ):
    script = []

    script.append(f"#!/bin/bash")
    script.append(f"#SBATCH --job-name={job_name}")
    script.append(f"#SBATCH --output=output%j.txt")
    script.append(f"#SBATCH --error=error%j.txt")
    script.append(f"#SBATCH --time={time}")  # dd-hh:mm:ss
    script.append(f"#SBATCH --partition={partition}")
    script.append(f"#SBATCH --nodes={n_nodes}")
    # script.append(f"#SBATCH --ntasks=1")
    script.append(f"#SBATCH --cpus-per-task={n_cpus}")
    script.append(f"#SBATCH --mem={mem}")
    if n_gpus > 0:
        script.append(f"#SBATCH --gres=gpu:{n_gpus}")
    if exclude_nodes is not None:
        script.append(f"#SBATCH -x {exclude_nodes}")

    script.append(f"{commands}")

    final_script = '\n'.join(script)

    if script_name is not None:
        with open(script_name, 'w') as out_script_file:
            out_script_file.write(final_script)
            
    return final_script