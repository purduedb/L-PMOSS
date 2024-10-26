import subprocess
from concurrent.futures import ThreadPoolExecutor


wl_list = [11, 11, 11, 11, 11, 11, 11, 11]
ecfg_list = [1, 3, 31, 41, 100, 101, 33, 7]
sidx_list = [204, 205, 206, 207, 208, 209, 210, 211]
rtg_list = [2, 2, 2, 2, 2, 2, 2, 2]


# Function to execute commands
def run_command(wl, ecfg, sidx, rtg):
    command = f"python run_dt_place.py --wl {wl} --ecfg {ecfg} --sidx {sidx} --rtg {rtg}"
    
    # Execute the command
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    
    # Print the output and any errors
    print(f"Running command: {command}")
    if result.returncode == 0:
        print(f"Success: {result.stdout}")
    else:
        print(f"Error: {result.stderr}")

# Using ThreadPoolExecutor to manage threads
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(run_command, wl, ecfg, sidx, rtg)
        for wl, ecfg, sidx, rtg in zip(wl_list, ecfg_list, sidx_list, rtg_list)
    ]

# Wait for all futures to complete
for future in futures:
    future.result()  # This will raise exceptions if the command failed

print("All commands have been executed.")
