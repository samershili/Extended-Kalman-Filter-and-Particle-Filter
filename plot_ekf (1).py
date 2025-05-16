import subprocess
import numpy as np
import matplotlib.pyplot as plt

def run_ekf_trial(data_noise, filter_noise, seed):
    cmd = [
        'python', 'localization.py', 'ekf',
        '--data-factor', str(data_noise),
        '--filter-factor', str(filter_noise),
        '--seed', str(seed)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        out = result.stdout
        err = float(out.split("Mean position error:")[1].split("\n")[0])
        anees = float(out.split("ANEES:")[1])
        return err, anees
    except Exception as e:
        print("Parsing error:", e)
        print(out)
        return None, None

def run_ekf_experiments(lock_data_noise):
    r_vals = [1/64, 1/16, 1/4, 1, 4, 16, 64]
    trials = 10
    errors, anees_vals = [], []

    for r in r_vals:
        e_list, a_list = [], []
        for t in range(trials):
            d_noise = 1 if lock_data_noise else r
            f_noise = r
            e, a = run_ekf_trial(d_noise, f_noise, t)
            if e is not None:
                e_list.append(e)
                a_list.append(a)
        errors.append(np.mean(e_list))
        anees_vals.append(np.mean(a_list))
    return r_vals, errors, anees_vals

print("[EKF] Experiment 3(b): Vary Data & Filter")
r1, err_b, anees_b = run_ekf_experiments(False)

print("[EKF] Experiment 3(c): Fixed Data, Vary Filter")
r2, err_c, anees_c = run_ekf_experiments(True)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(r1, err_b, 'o-', label='Vary Data & Filter')
plt.plot(r2, err_c, 's--', label='Filter Only')
plt.xscale('log')
plt.xlabel('Noise Scale r')
plt.ylabel('Mean Position Error')
plt.title('EKF Position Error')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(r1, anees_b, 'o-', label='Vary Data & Filter')
plt.plot(r2, anees_c, 's--', label='Filter Only')
plt.xscale('log')
plt.xlabel('Noise Scale r')
plt.ylabel('ANEES')
plt.title('EKF ANEES')
plt.legend()

plt.tight_layout()
plt.show()