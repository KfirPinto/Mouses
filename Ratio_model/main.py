import argparse
import yaml
import sys
import os
import warnings

# Suppress deprecation warnings from dependencies
warnings.filterwarnings('ignore')

# Ensure we can import from src/ regardless of where script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from src.data_loader import load_and_prep_data
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Modular Ratio Model Runner")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to config YAML (default: config.yaml in Ratio_model folder)")
    parser.add_argument("--hyper", action="store_true", help="Run hyperparameter search instead of single run")
    args = parser.parse_args()

    # Handle both relative and absolute paths
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        # Look in Ratio_model folder if relative path given
        config_path = os.path.join(script_dir, args.config)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        print(f"Please provide a valid config path.")
        print(f"Available configs in {script_dir}/configs/:")
        try:
            configs = os.listdir(os.path.join(script_dir, "configs"))
            for c in configs:
                print(f"  - {c}")
        except:
            pass
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"--- Starting Run: {config['output_settings']['model_name']} ---")
    print(f"Config: {config_path}")

    # Make data paths relative to Mouses folder (parent of Ratio_model)
    mouses_dir = os.path.dirname(script_dir)
    config['data']['censored_path'] = os.path.join(mouses_dir, config['data']['censored_path'])
    config['data']['uncensored_path'] = os.path.join(mouses_dir, config['data']['uncensored_path'])
    
    # Make output paths relative to Mouses folder
    config['output_settings']['base_folder'] = os.path.join(mouses_dir, config['output_settings']['base_folder'])

    # 2. Load Data
    censored, uncensored = load_and_prep_data(
        config['data']['censored_path'],
        config['data']['uncensored_path'],
        config['data'].get('age_filter')
    )
    
    # Pass data to config dict
    config['data_loaded'] = (censored, uncensored)

    # 3. Execute Pipeline
    run_pipeline(config, run_hyper=args.hyper)

if __name__ == "__main__":
    main()