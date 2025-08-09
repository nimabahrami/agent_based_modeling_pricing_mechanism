import os
import sys
import subprocess
import shutil
from datetime import datetime

def run_simulation_with_cove_markup(cove_markup):
    
    output_dir = f'COVE_factor/cove_markup_{cove_markup:.1f}'
    os.makedirs(output_dir, exist_ok=True)
    
    env = os.environ.copy()
    env['COVE_MARKUP'] = str(cove_markup)
    
    try:
        result = subprocess.run([sys.executable, 'run_simulation.py'], 
                              env=env, 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f" Simulation completed for COVE markup {cove_markup}")
            
            if os.path.exists('simulation_results'):
                for file in os.listdir('simulation_results'):
                    src = os.path.join('simulation_results', file)
                    dst = os.path.join(output_dir, file)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    elif os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"Results copied to {output_dir}")
                
                for file in os.listdir('simulation_results'):
                    src = os.path.join('simulation_results', file)
                    if os.path.isfile(src):
                        os.remove(src)
                    elif os.path.isdir(src):
                        shutil.rmtree(src)
                print(f"cleaned folder")
            
            return True
        else:
            print(f"✗ Simulation failed for COVE markup {cove_markup}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error for COVE markup {cove_markup}: {e}")
        return False

def main():
    print("Energy Community Model - COVE Markup Factor Analysis")
    print("="*60)
    
    cove_markup_factors = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    
    print(f"Testing COVE markup factors: {cove_markup_factors}")
    print(f"Total simulations to run: {len(cove_markup_factors)}")
    
    os.makedirs('COVE_factor', exist_ok=True)
    
    successful_runs = []
    failed_runs = []
    
    try:
        for cove_markup in cove_markup_factors:
            success = run_simulation_with_cove_markup(cove_markup)
            if success:
                successful_runs.append(cove_markup)
            else:
                failed_runs.append(cove_markup)
        
        print(f"{'='*60}")
        print(f"Successful runs: {len(successful_runs)}")
        print(f"Failed runs: {len(failed_runs)}")
        
        if successful_runs:
            print(f"Successful COVE markup factors: {successful_runs}")
        if failed_runs:
            print(f"Failed COVE markup factors: {failed_runs}")
        
        print(f"\nResults saved in: COVE_factor/")
        
        return successful_runs, failed_runs
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return successful_runs, failed_runs
    except Exception as e:
        print(f"\nError in main execution: {e}")
        return successful_runs, failed_runs

if __name__ == "__main__":
    successful_runs, failed_runs = main() 
