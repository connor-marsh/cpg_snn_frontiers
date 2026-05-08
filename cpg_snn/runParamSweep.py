import os
import time
import numpy as np
params = ["hidden", "seq_len", "beta", "n_layers"]
ranges = [np.arange(20, 200, 10), np.arange(1, 30, 2), np.arange(0.3, 1.0001, 0.05), np.arange(1, 10, 1)]
# Specifically mess with beta range because np.arange doesnt make an easy way to set this otherwise
ranges[params.index("beta")][-1]=0.999

dir_prefix = "vishnu_multigait_stuff/"
for param, range in zip(params, ranges):
    out_dir = dir_prefix + "param_sweep_" + param
    os.system("mkdir " + out_dir)
    print("Creating runs directory: " + out_dir)
    print("###########################################\n###########################################")
    print("Running Param Sweep on param:", param, "with range:", (str(min(range)) + " - " + str(max(range))))
    print("###########################################")
    runTimes = []
    allRmseMats = []
    for value in range:
        print("Running param: ", param, "with value: ", value)
        foldername = dir_prefix + "sweep_outputs/outputs_cpg_mixing_" + param + "_" + str(value)
        startTime = time.perf_counter()
        os.system("python3 vishnu_multigait_stuff/cpg_snn_torch_multi_gait_mixing.py --" + param + " " + str(value) + " --epochs 500 --out_dir " + foldername)
        runTimes.append(time.perf_counter()-startTime)
        allRmseMats.append(np.load(foldername+"/rmse_matrices.npy"))
        print("Done!")
        time.sleep(1) # Wait a second to make it easier to CTRL+C out
        print("###########################################\n###########################################")
        
    np.save(out_dir+"/x_axis", range)
    np.save(out_dir+"/all_rmse_matrices", np.array(allRmseMats))
    np.save(out_dir+"/run_time", np.array(runTimes))