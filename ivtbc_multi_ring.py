import os, subprocess, shutil, json, re
import numpy as np

r1        = 0.00133
r2        = 0.00334
T_inf     = 791.0
gap       = 124.0
hp_radius = 0.0127
hp_length = 1.60
hp_area   = 2 * np.pi * hp_radius * hp_length
hp_length_slice = 0.0114   # actual mesh axial extent, per checkMesh — used ONLY for wall-flux conversion, not for cell_vol
hp_area_slice = 2 * np.pi * hp_radius * hp_length_slice
epsilon   = 0.1
omega     = 0.5
max_iter  = 20
case_dir  = os.path.expanduser('~/HPR_case')

ring_data = {
    'ring_0': {'r_cm': 2.65,  'count': 6,   'q_hp_W': 4811},
    'ring_1': {'r_cm': 5.30,  'count': 36,  'q_hp_W': 5611},
    'ring_2': {'r_cm': 7.95,  'count': 72,  'q_hp_W': 8034},
    'ring_3': {'r_cm': 10.60, 'count': 108, 'q_hp_W': 9754},
    'ring_4': {'r_cm': 13.25, 'count': 144, 'q_hp_W': 3925},
    'ring_5': {'r_cm': 15.90, 'count': 180, 'q_hp_W': 248},
    'ring_6': {'r_cm': 18.55, 'count': 216, 'q_hp_W': 126},
}

def update_heat_source(case_dir, q_vol):
    f = os.path.join(case_dir, 'constant', 'solid', 'fvModels')
    with open(f) as fh: content = fh.read()
    content = re.sub(r'q\s+[\d.e+]+;', f'q               {q_vol:.4e};', content)
    with open(f, 'w') as fh: fh.write(content)

def update_BC(case_dir, Tvap):
    T_file = os.path.join(case_dir, '0', 'solid', 'T')
    with open(T_file) as f: content = f.read()
    content = re.sub(r'Ta\s+uniform\s+[\d.]+;',
                     f'Ta              uniform {Tvap:.4f};', content)
    content = re.sub(r'internalField\s+uniform\s+[\d.]+;',
                     f'internalField   uniform {Tvap:.4f};', content)
    content = re.sub(r'value\s+uniform\s+[\d.]+;',
                     f'value           uniform {Tvap:.4f};', content)
    with open(T_file, 'w') as f: f.write(content)
    print(f'  Updated Ta = {Tvap:.4f} K')

def read_Q(case_dir):
    flux_file = os.path.join(case_dir, 'postProcessing', 'solid',
                             'wallHeatFlux', '0', 'wallHeatFlux.dat')
    with open(flux_file) as f:
        lines = [l for l in f.readlines()
                 if not l.startswith('#') and l.strip()]
    return abs(float(lines[-1].split()[4]))

def run_openfoam(case_dir):
    result = subprocess.run(['foamMultiRun'], cwd=case_dir,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print('  OpenFOAM failed')
        print(result.stderr[-200:])
        return False
    return True

def clean_times(case_dir):
    for entry in os.listdir(case_dir):
        try:
            t = float(entry)
            if t > 0:
                shutil.rmtree(os.path.join(case_dir, entry))
        except ValueError:
            continue

if __name__ == '__main__':
    print('=' * 65)
    print('MULTI-RING IVTBC — Design v2 (r_HP=12.7mm, N=762)')
    print('=' * 65)

    all_results = {}
    cell_vol = (0.055**2) * 1.60   # full pipe length — averages density along the axial direction, correct even though the mesh only spans a slice

    for ring_name, ring in ring_data.items():
        q_hp_W = ring['q_hp_W']
        q_vol  = q_hp_W / cell_vol   # single pipe per unit-cell domain (confirmed: blockMeshDict inner square = 1 heat pipe)
        Tvap_analytical = r2 * (q_hp_W/hp_area) + T_inf + gap

        print(f'\n--- {ring_name} | r={ring["r_cm"]}cm | q={q_hp_W}W/pipe ---')
        print(f'    q_vol = {q_vol:.3e} W/m3')
        print(f'    Tvap (analytical) = {Tvap_analytical:.2f} K ({Tvap_analytical-273.15:.2f} C)')

        # Initialize from analytical Tvap
        Tvap = Tvap_analytical
        converged = False
        iteration = 0
        ring_results = []

        update_heat_source(case_dir, q_vol)

        while not converged and iteration < max_iter:
            update_BC(case_dir, Tvap)
            clean_times(case_dir)
            ok = run_openfoam(case_dir)
            if not ok:
                print(f'  OpenFOAM failed at iter {iteration}')
                break
            Q_wall = read_Q(case_dir)
            q_flux = Q_wall / hp_area_slice   # corrected: match to the slice wall area actually measured, not the full-pipe area
            Tvap_raw = r2 * q_flux + T_inf + gap
            Tvap_new = omega * Tvap_raw + (1-omega) * Tvap
            diff = abs(Tvap_new - Tvap)
            print(f'  iter {iteration}: Q={Q_wall:.2f}W Tvap={Tvap_new:.2f}K ({Tvap_new-273.15:.2f}C) diff={diff:.4f}K')
            ring_results.append({'iter': iteration, 'Q': Q_wall,
                                  'Tvap': Tvap_new, 'diff': diff})
            if diff < epsilon:
                converged = True
                print(f'  CONVERGED')
            Tvap = Tvap_new
            iteration += 1

        Tm = Tvap + r1 * (ring['q_hp_W']/hp_area)
        all_results[ring_name] = {
            'r_cm': ring['r_cm'],
            'count': ring['count'],
            'q_hp_W': ring['q_hp_W'],
            'Tvap_analytical_K': Tvap_analytical,
            'Tvap_analytical_C': Tvap_analytical - 273.15,
            'Tvap_IVTBC_K': Tvap,
            'Tvap_IVTBC_C': Tvap - 273.15,
            'Tm_K': Tm,
            'Tm_C': Tm - 273.15,
            'converged': converged,
            'iterations': iteration,
            'safe': bool(Tvap < 1173 and Tm < 1600)
        }
        print(f'  RESULT: Tvap = {Tvap:.4f} K ({Tvap-273.15:.4f} C)')

    print('\n' + '='*65)
    print('FINAL PER-RING IVTBC RESULTS')
    print('='*65)
    print(f"{'Ring':<8} {'Tvap_anal(C)':<14} {'Tvap_IVTBC(C)':<15} {'Diff(K)':<10} {'Safe?'}")
    print('-'*55)
    for rname, r in all_results.items():
        diff = abs(r['Tvap_IVTBC_K'] - r['Tvap_analytical_K'])
        print(f"{rname:<8} {r['Tvap_analytical_C']:<14.2f} {r['Tvap_IVTBC_C']:<15.2f} {diff:<10.2f} {'✅' if r['safe'] else '❌'}")

    out = os.path.expanduser(
        '~/openfoam-cases/Heat-Pipe-Reactor/PHASE 3/ivtbc_fullwedge_perring.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved: {out}')
