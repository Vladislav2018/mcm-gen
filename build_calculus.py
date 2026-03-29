import json
import sympy as sp
import numpy as np
import traceback
import sys
import os
import multiprocessing
import uuid
from src.sampler import TaskExporter, determine_domain
from upgrade_manuals4 import get_points, adaptive_sample

def load_jsonl(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data

def save_jsonl(filepath, data):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')

def process_worker_dict(args):
    """Worker function for multiprocessing returning dict directly."""
    # Move imports here to ensure they exist in worker process
    from src.sampler import TaskExporter, determine_domain
    from upgrade_manuals4 import get_points, adaptive_sample
    import sympy as sp
    
    expr_str, config_dict, task_id = args
    x = sp.Symbol('x', real=True)
    try:
        expr = sp.parse_expr(expr_str, local_dict={'x': x})
            
        results = {}
        # 1. Derivative
        try:
            deriv_expr = sp.diff(expr, x)
            deriv_expr = sp.simplify(deriv_expr)
            d_str = str(deriv_expr).replace(" ", "")
            if d_str != "0":
                # Calculate points inline
                pts = adaptive_sample(d_str, n=25)
                pts_success, x_vals, y_vals = get_points(deriv_expr, custom_x=pts, n=25)
                if not pts_success:
                    pts_success, x_vals, y_vals = get_points(deriv_expr, custom_x=None, n=25)
                if pts_success:
                    results['derivative'] = {'str': d_str, 'x': x_vals, 'y': y_vals}
        except Exception as e:
            print(f"  [DERIV FAIL] {expr_str}: {e}", flush=True)
            
        # 2. Integral
        try:
            integ_expr = sp.integrate(expr, x)
            integ_expr = sp.simplify(integ_expr)
            if not ('Integral' in str(integ_expr) or integ_expr.has(sp.Integral)):
                i_str = str(integ_expr).replace(" ", "")
                if x in integ_expr.free_symbols:
                    pts = adaptive_sample(i_str, n=25)
                    pts_success, x_vals, y_vals = get_points(integ_expr, custom_x=pts, n=25)
                    if not pts_success:
                        pts_success, x_vals, y_vals = get_points(integ_expr, custom_x=None, n=25)
                    if pts_success:
                        results['integral'] = {'str': i_str, 'x': x_vals, 'y': y_vals}
        except Exception as e:
            print(f"  [INTEG FAIL] {expr_str}: {e}", flush=True)
            
        return results
    except Exception as e:
        return {'error': str(e)}

def main():
    source_file = "data/benchmark_tasks.jsonl"
    deriv_file = "data/benchmark_derivatives.jsonl"
    integ_file = "data/benchmark_integrals.jsonl"
    
    # Empty files if they exist
    open(deriv_file, 'w').close()
    open(integ_file, 'w').close()
    
    tasks = load_jsonl(source_file)
    print(f"Loaded {len(tasks)} tasks from {source_file}")
    
    # Prepare arguments for multiprocessing
    args_list = []
    for task in tasks:
        expr_str = task['ground_truth']['formula']
        config_dict = task['complexity_vector']
        args_list.append((expr_str, config_dict, task['task_id']))
        
    seen_derivs = set()
    seen_integs = set()
    
    deriv_count = 0
    integ_count = 0
    
    x_sym = sp.Symbol('x', real=True)
    uncompleted = args_list.copy()
    total_tasks = len(args_list)
    processed = 0
    
    while uncompleted:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        futures = []
        for args in uncompleted:
            res_obj = pool.apply_async(process_worker_dict, (args,))
            futures.append((args, res_obj))
            
        new_uncompleted = []
        timeout_occurred = False
        
        for idx, (args, res_obj) in enumerate(futures):
            expr_str = args[0]
            
            if timeout_occurred:
                new_uncompleted.append(args)
                continue
                
            if processed % 10 == 0:
                print(f"Processed {processed}/{total_tasks} formulas... (D: {deriv_count}, I: {integ_count})", flush=True)
                
            try:
                res = res_obj.get(timeout=30.0)
                processed += 1
                if res is None or 'error' in res:
                    continue
            except multiprocessing.TimeoutError:
                print(f"TIMEOUT: Function {expr_str} took too long to derive/integrate. Restarting pool.", flush=True)
                processed += 1
                timeout_occurred = True
                pool.terminate()
                pool.join()
                continue
                
            class MockConfig:
                def __init__(self, c):
                    self.a = c['a']; self.b = c['b']; self.c = c['c']
            config = MockConfig(args[1])
            
            # Handle Derivative
            if 'derivative' in res:
                d_data = res['derivative']
                if d_data['str'] not in seen_derivs:
                    seen_derivs.add(d_data['str'])
                    try:
                        d_expr = sp.parse_expr(d_data['str'], local_dict={'x': x_sym})
                        meta = {"singularities": [], "is_periodic": False, "domain": "R"}
                        task = TaskExporter.create_task(d_expr, d_data['x'], d_data['y'], config, meta)
                        task['source'] = 'derivative_of_' + expr_str
                        save_jsonl(deriv_file, task)
                        deriv_count += 1
                    except Exception:
                        pass

            # Handle Integral
            if 'integral' in res:
                i_data = res['integral']
                if i_data['str'] not in seen_integs:
                    seen_integs.add(i_data['str'])
                    try:
                        i_expr = sp.parse_expr(i_data['str'], local_dict={'x': x_sym})
                        meta = {"singularities": [], "is_periodic": False, "domain": "R"}
                        task = TaskExporter.create_task(i_expr, i_data['x'], i_data['y'], config, meta)
                        task['source'] = 'integral_of_' + expr_str
                        save_jsonl(integ_file, task)
                        integ_count += 1
                    except Exception:
                        pass

        if not timeout_occurred:
            pool.close()
            pool.join()
            
        uncompleted = new_uncompleted
    
    print(f"\nDone! Generated {deriv_count} unique valid derivatives into {deriv_file}")
    print(f"Generated {integ_count} unique valid integrals into {integ_file}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
