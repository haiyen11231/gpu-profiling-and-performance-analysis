import os
import subprocess
import itertools
import csv
import re
import argparse

# --- Setup Argument Parsing for Distributed Runs ---
parser = argparse.ArgumentParser(description="Run a distributed grid sweep for GPU benchmarking.")
parser.add_argument("--pod", type=int, default=0, help="ID of current pod (e.g., 0, 1...)")
parser.add_argument("--total_pods", type=int, default=1, help="Total number of pods running the sweep")
args = parser.parse_args()

# --- HYPERPARAMETER SEARCH SPACE ---
block_sizes = [8, 16, 32]
reg_tiles = [1, 2, 4]
unroll_factors = ["1", "4", "256", "0"] 
vec_types = ["float", "float2", "float4"]
num_streams = [1, 2, 4, 8]

source_file = "rocmGrail.cpp"
executable = f"./run_gemm_pod_{args.pod}" # Unique executable name per pod to prevent locking issues
output_csv = f"results_pod_{args.pod}.csv"

# Regex to extract execution time from C++ output
time_regex = re.compile(r"Total Time:\s+([0-9.]+)\s+ms")

def run_grid_sweep():
    # Generate all possible combinations
    all_combinations = list(itertools.product(block_sizes, reg_tiles, unroll_factors, vec_types, num_streams))
    
    # Filter combinations specific to this pod using modulo sharding
    pod_combinations = [c for i, c in enumerate(all_combinations) if i % args.total_pods == args.pod]
    total_runs = len(pod_combinations)
    
    print(f"Starting Grid Sweep on Pod {args.pod}/{args.total_pods - 1}")
    print(f"Assigned {total_runs} out of {len(all_combinations)} total configurations.\n" + "="*50)

    # Setup CSV Writer
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Block_Size", "Reg_Tile", "Unroll", "Vec_Type", "Streams", 
                         "V0", "V1", "V2", "V3", "V4"])
        
        for idx, (bs, rt, uf, vt, ns) in enumerate(pod_combinations):
            # Formatted unroll string for printing clarity
            uf_print = "Compiler Default" if uf == "0" else ("Full" if uf == "256" else uf)
            print(f"[{idx+1}/{total_runs}] Compiling config: BS={bs}, RT={rt}, UF={uf_print}, VEC={vt}, Streams={ns}")
            
            # 1. Inject Macros into HIPCC Compilation
            compile_cmd = [
                "nvcc", source_file, "-o", executable, "-O3",
                f"-DBLOCK_SIZE_X={bs}", f"-DREG_TILE_X={rt}",
                f"-DUF={uf}", f"-DVEC_TYPE={vt}", f"-DNUM_STREAMS={ns}"
            ]
            
            try:
                subprocess.run(compile_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"  -> COMPILE FAILED! Skipping. Error:\n{e.stderr.decode('utf-8')}")
                # Ensure we write exactly 10 columns for a clean CSV
                writer.writerow([bs, rt, uf, vt, ns, "COMPILE_ERROR", "COMPILE_ERROR", "COMPILE_ERROR", "COMPILE_ERROR", "COMPILE_ERROR"])
                continue
            
            # 2. Execute the compiled kernel
            try:
                result = subprocess.run([executable], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = result.stdout
            except subprocess.CalledProcessError as e:
                output = e.stdout

            # 3. Parse the results for Versions 0 through 4
            times = {"0": "NaN", "1": "NaN", "2": "NaN", "3": "NaN", "4": "NaN"}
            current_v = None
            
            for line in output.split('\n'):
                if "Starting benchmark for Version" in line:
                    current_v = re.search(r"Version (\d)", line).group(1)
                elif "RESULT: OUT_OF_RESOURCES" in line:
                    times[current_v] = "NaN (OOM)"
                elif "Total Time:" in line:
                    match = time_regex.search(line)
                    if match:
                        times[current_v] = match.group(1)
            
            # 4. Write to CSV
            print(f"  -> Results: V0={times['0']}ms, V1={times['1']}ms, V2={times['2']}ms, V3={times['3']}ms, V4={times['4']}ms")
            writer.writerow([bs, rt, uf, vt, ns, times["0"], times["1"], times["2"], times["3"], times["4"]])

            # Clean up the executable (optional)
            if os.path.exists(executable):
                os.remove(executable)

    print("\n" + "="*50)
    print(f"Grid Sweep Complete on Pod {args.pod}! Results saved to {output_csv}")

if __name__ == "__main__":
    run_grid_sweep()