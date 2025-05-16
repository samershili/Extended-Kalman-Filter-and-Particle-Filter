import subprocess
import numpy as np
import matplotlib.pyplot as plt

def run_pf_trial(data_noise, filter_noise, seed, n_particles):
    cmd = [
        'python', 'localization.py', 'pf',
        '--data-factor', str(data_noise),
        '--filter-factor', str(filter_noise),
        '--seed', str(seed),
        '--num-particles', str(n_particles)
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

def run_pf_experiments(lock_data_noise, n_particles):
    r_vals = [1/64, 1/16, 1/4, 1, 4, 16, 64]
    trials = 10
    errors, anees_vals = [], []

    for r in r_vals:
        e_list, a_list = [], []
        for t in range(trials):
            d_noise = 1 if lock_data_noise else r
            f_noise = r
            e, a = run_pf_trial(d_noise, f_noise, t, n_particles)
            if e is not None:
                e_list.append(e)
                a_list.append(a)
        errors.append(np.mean(e_list))
        anees_vals.append(np.mean(a_list))
    return r_vals, errors, anees_vals

print("[PF] 4(b): Vary Data & Filter")
r1, err_b, anees_b = run_pf_experiments(False, 100)

print("[PF] 4(c): Filter Only")
r2, err_c, anees_c = run_pf_experiments(True, 100)

particles = [20, 50, 500]
err_particles = {}

for n in particles:
    print(f"[PF] 4(d): {n} particles")
    r, e, _ = run_pf_experiments(True, n)
    err_particles[n] = e

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(r1, err_b, 'o-', label='Vary Data & Filter')
plt.plot(r2, err_c, 's--', label='Filter Only')
plt.xscale('log')
plt.xlabel('Noise Scale r')
plt.ylabel('Mean Error')
plt.title('PF Mean Position Error')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(r1, anees_b, 'o-', label='Vary Data & Filter')
plt.plot(r2, anees_c, 's--', label='Filter Only')
plt.xscale('log')
plt.xlabel('Noise Scale r')
plt.ylabel('ANEES')
plt.title('PF ANEES')
plt.legend()

plt.subplot(1, 3, 3)
for n in particles:
    plt.plot(r, err_particles[n], label=f'{n} particles')
plt.xscale('log')
plt.xlabel('Noise Scale r')
plt.ylabel('Mean Error')
plt.title('Error vs Particles')
plt.legend()

plt.tight_layout()
plt.show()
