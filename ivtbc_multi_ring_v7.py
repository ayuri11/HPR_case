import os, subprocess, shutil, json, re, time
import numpy as np

# ---------------- model constants (unchanged) ----------------
r1        = 0.00133
r2        = 0.00334
T_inf     = 791.0
gap       = 124.0
hp_radius = 0.0127
hp_length = 1.60
hp_area   = 2 * np.pi * hp_radius * hp_length            # full-pipe wall area
hp_length_slice = 0.0114                                  # mesh axial extent
hp_area_slice   = 2 * np.pi * hp_radius * hp_length_slice # wall area of the meshed slice
slice_frac      = hp_length_slice / hp_length             # slice's share of one pipe's power
V_GRAPHITE = 2.8714584e-05   # m3, graphite volume of the REBUILT 12.7 mm solver mesh (checkMesh -region solid)
T_CLAD_K  = 900.0  + 273.15
T_TRISO_K = 1600.0 + 273.15
epsilon, omega, max_iter = 0.1, 0.5, 20
STEADY_TOL = 0.02            # wall heat flow must be within 2% of injected power
END_TIMES  = (600, 1200, 2400)   # tried in turn until steady
DELTA_T    = 1
case_dir  = os.path.expanduser('~/HPR_case')

ring_data = {
    'ring_0': {'r_cm': 2.65,  'count': 6,   'q_hp_W': 21688.7},
    'ring_1': {'r_cm': 8.15,  'count': 36,  'q_hp_W': 20992.1},
    'ring_2': {'r_cm': 13.65, 'count': 72,  'q_hp_W': 19348.4},
    'ring_3': {'r_cm': 19.15, 'count': 126, 'q_hp_W': 1754.5},
}

def update_heat_source(case_dir, q_vol):
    f = os.path.join(case_dir, 'constant', 'solid', 'fvModels')
    with open(f) as fh: content = fh.read()
    content = re.sub(r'q\s+[\d.e+]+;', f'q               {q_vol:.4e};', content)
    with open(f, 'w') as fh: fh.write(content)

def update_BC(case_dir, Tvap, T_init):
    """Ta and wall value = Tvap; internal field starts near the expected steady state."""
    T_file = os.path.join(case_dir, '0', 'solid', 'T')
    with open(T_file) as f: content = f.read()
    content = re.sub(r'Ta\s+uniform\s+[\d.]+;', f'Ta              uniform {Tvap:.4f};', content)
    content = re.sub(r'internalField\s+uniform\s+[\d.]+;', f'internalField   uniform {T_init:.4f};', content)
    content = re.sub(r'value\s+uniform\s+[\d.]+;', f'value           uniform {Tvap:.4f};', content)
    with open(T_file, 'w') as f: f.write(content)
    print(f'  Ta = {Tvap:.4f} K, initial T = {T_init:.2f} K')

def set_controls(case_dir, end_time, delta_t):
    f = os.path.join(case_dir, 'system', 'controlDict')
    with open(f) as fh: c = fh.read()
    c = re.sub(r'(?m)^(\s*endTime\s+)[^;]+;', lambda m: f'{m.group(1)}{end_time};', c)
    c = re.sub(r'(?m)^(\s*deltaT\s+)[^;]+;', lambda m: f'{m.group(1)}{delta_t};', c)
    with open(f, 'w') as fh: fh.write(c)

def read_Q(case_dir):
    flux_file = os.path.join(case_dir, 'postProcessing', 'solid', 'wallHeatFlux', '0', 'wallHeatFlux.dat')
    with open(flux_file) as f:
        lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]
    return abs(float(lines[-1].split()[4]))

def latest_time_dir(case_dir):
    best, bt = None, 0.0
    for e in os.listdir(case_dir):
        try: t = float(e)
        except ValueError: continue
        if t > bt and os.path.isdir(os.path.join(case_dir, e)): best, bt = e, t
    return best

def read_T_stats(case_dir):
    d = latest_time_dir(case_dir)
    if d is None: return None
    try:
        with open(os.path.join(case_dir, d, 'solid', 'T')) as fh: c = fh.read()
        m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(\s*([^)]*)\)', c, re.S)
        if not m: return None
        vals = [float(x) for x in m.group(2).split()]
        return {'max': max(vals), 'mean': sum(vals)/len(vals), 'n': len(vals)}
    except Exception as e:
        print('  (could not read T field:', e, ')'); return None

def run_openfoam(case_dir):
    r = subprocess.run(['foamMultiRun'], cwd=case_dir, capture_output=True, text=True)
    if r.returncode != 0:
        print('  OpenFOAM failed'); print(r.stderr[-300:]); return False
    return True

def clean_times(case_dir):
    for entry in os.listdir(case_dir):
        try:
            if float(entry) > 0: shutil.rmtree(os.path.join(case_dir, entry))
        except ValueError:
            continue

def solve_to_steady(case_dir, injected_W):
    """Run until wall heat flow = injected power (energy balance), extending endTime if needed."""
    res = None
    for et in END_TIMES:
        set_controls(case_dir, et, DELTA_T)
        clean_times(case_dir)
        pp = os.path.join(case_dir, 'postProcessing', 'solid', 'wallHeatFlux')
        if os.path.isdir(pp): shutil.rmtree(pp)
        t0 = time.time()
        if not run_openfoam(case_dir): return None
        Q = read_Q(case_dir); ratio = Q / injected_W
        res = {'Q': Q, 'ratio': ratio, 'end_time': et, 'steady': abs(ratio-1) <= STEADY_TOL,
               'T': read_T_stats(case_dir), 'secs': time.time()-t0}
        print(f'    endTime={et}s: wall heat {Q:.3f} W / injected {injected_W:.3f} W = {ratio:.4f}  ({res["secs"]:.0f}s)')
        if res['steady']: return res
    print('  !! WARNING: energy balance not closed - this result is NOT reliable')
    return res

if __name__ == '__main__':
    print('=' * 70)
    print('MULTI-RING IVTBC v6 - steady-state, 12.7 mm mesh, energy-balance checked')
    print('=' * 70)
    lock = os.path.join(case_dir, '.ivtbc_v6.lock')
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip()); os.kill(pid, 0)
            raise SystemExit(f'Another IVTBC run is active (PID {pid}). Stop it first: kill {pid}')
        except (ValueError, ProcessLookupError):
            pass   # stale lock
    with open(lock, 'w') as fh: fh.write(str(os.getpid()))
    cd = os.path.join(case_dir, 'system', 'controlDict')
    shutil.copy(cd, cd + '.pre_v6')
    all_results = {}
    try:
        for ring_name, ring in ring_data.items():
            q_hp_W = ring['q_hp_W']
            q_design = q_hp_W / hp_area                       # design wall flux, W/m2
            injected = q_hp_W * slice_frac                    # W deposited in the meshed slice
            q_vol = injected / V_GRAPHITE                     # heat goes into graphite only
            Tvap_analytical = r2 * q_design + T_inf + gap
            cond_guess = 80.0 * q_hp_W / 21688.7              # rough start guess for conduction rise (K)
            print(f'\n--- {ring_name} | r={ring["r_cm"]}cm | q={q_hp_W} W/pipe ---')
            print(f'    q_vol = {q_vol:.4e} W/m3   injected = {injected:.3f} W/slice')
            print(f'    Tvap (energy-balance) = {Tvap_analytical:.2f} K ({Tvap_analytical-273.15:.2f} C)')
            update_heat_source(case_dir, q_vol)
            Tvap, converged, iteration, last = Tvap_analytical, False, 0, None
            all_steady = True
            while not converged and iteration < max_iter:
                update_BC(case_dir, Tvap, Tvap + r1*q_design + cond_guess)
                last = solve_to_steady(case_dir, injected)
                if last is None: break
                all_steady = all_steady and last['steady']
                q_flux = last['Q'] / hp_area_slice
                Tvap_raw = r2 * q_flux + T_inf + gap
                Tvap_new = omega * Tvap_raw + (1-omega) * Tvap
                diff = abs(Tvap_new - Tvap)
                print(f'  iter {iteration}: Tvap={Tvap_new:.2f}K ({Tvap_new-273.15:.2f}C) diff={diff:.4f}K')
                if diff < epsilon:
                    converged = True; print('  CONVERGED')
                Tvap = Tvap_new; iteration += 1
            Tm_formula = Tvap + r1 * q_design
            Tmax_cfd = last['T']['max'] if last and last.get('T') else None
            Tm_used = max(Tm_formula, Tmax_cfd) if Tmax_cfd else Tm_formula
            all_results[ring_name] = {
                'r_cm': ring['r_cm'], 'count': ring['count'], 'q_hp_W': q_hp_W,
                'Tvap_analytical_C': Tvap_analytical-273.15, 'Tvap_IVTBC_C': Tvap-273.15,
                'Tm_formula_C': Tm_formula-273.15,
                'Tmax_graphite_CFD_C': (Tmax_cfd-273.15) if Tmax_cfd else None,
                'energy_balance_ratio': last['ratio'] if last else None,
                'endTime_used_s': last['end_time'] if last else None,
                'converged': converged, 'iterations': iteration, 'steady_state_ok': bool(all_steady),
                'cladding_margin_K': T_CLAD_K - Tvap,
                'triso_margin_K_formula': T_TRISO_K - Tm_formula,
                'triso_margin_K_cfdmax': (T_TRISO_K - Tmax_cfd) if Tmax_cfd else None,
                'safe': bool(Tvap < T_CLAD_K and Tm_used < T_TRISO_K) if all_steady else None,
            }
            print(f'  RESULT: Tvap = {Tvap-273.15:.2f} C, Tm(formula) = {Tm_formula-273.15:.2f} C'
                  + (f', Tmax(CFD) = {Tmax_cfd-273.15:.2f} C' if Tmax_cfd else ''))
    finally:
        shutil.copy(cd + '.pre_v6', cd)   # restore original controlDict
        if os.path.exists(lock): os.remove(lock)

    print('\n' + '=' * 90)
    print('FINAL PER-RING IVTBC RESULTS (steady state)')
    print('=' * 90)
    print(f"{'Ring':<7}{'Tvap(C)':<10}{'clad mgn(K)':<13}{'Tm form(C)':<12}{'Tmax CFD(C)':<13}{'TRISO mgn(K)':<14}{'EB ratio':<10}{'Safe?'}")
    for rn, r in all_results.items():
        tmx = f"{r['Tmax_graphite_CFD_C']:.1f}" if r['Tmax_graphite_CFD_C'] else 'n/a'
        trm = r['triso_margin_K_cfdmax'] if r['triso_margin_K_cfdmax'] is not None else r['triso_margin_K_formula']
        eb  = f"{r['energy_balance_ratio']:.3f}" if r['energy_balance_ratio'] else 'n/a'
        sf  = {True:'YES', False:'NO', None:'UNRELIABLE'}[r['safe']]
        print(f"{rn:<7}{r['Tvap_IVTBC_C']:<10.1f}{r['cladding_margin_K']:<13.1f}{r['Tm_formula_C']:<12.1f}{tmx:<13}{trm:<14.1f}{eb:<10}{sf}")
    out = os.path.expanduser('~/openfoam-cases/Heat-Pipe-Reactor/PHASE 3/ivtbc_fullwedge_perring_v7.json')
    with open(out, 'w') as f: json.dump(all_results, f, indent=2)
    print(f'\nSaved: {out}')
