import numpy as np
import os
import pprint

NPY_PATH = "/home/dvrkteam/i4h-workflows/workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/psm_reach/2026-01-20_11-33-32/env_agent_data.npy"

def inspect_metadata():
    if not os.path.exists(NPY_PATH):
        print(f"ERROR: Target file not found at:\n{NPY_PATH}")
        return

    data = np.load(NPY_PATH, allow_pickle=True).item()

    print("\n" + "="*80)
    print("SYSTEM CONFIGURATION AND OBSERVATION DATA REPORT")
    print("="*80)

    # Section 1: Observations (Ora dentro 'data')
    print("\nPART 1: OBSERVATION SPECIFICATIONS")
    print("-" * 50)
    # Cambiato da .get('metadata') a .get('data')
    obs_info = data.get('data', {}).get('obs_info', {})
    print(f"{'Key':<15} | {'Shape':<22} | {'Dtype':<15}")
    print("-" * 50)
    for key, info in obs_info.items():
        print(f"{key:<15} | {str(info['shape']):<22} | {info['dtype']}")

    # Section 2: General Info e Classi
    print("\nPART 2: AGENT AND ENVIRONMENT SUMMARY")
    print("-" * 50)
    main_data = data.get('data', {})
    print(f"Algorithm Class:  {main_data.get('algo_class', 'N/A')}")
    print(f"Policy Class:     {main_data.get('policy_class', 'N/A')}")
    print(f"Num Environments: {main_data.get('num_envs', 'N/A')}")
    print(f"Action Dimension: {main_data.get('num_actions', 'N/A')}")

    # Section 3: Hyperparameters
    print("\nPART 3: TRAINING HYPERPARAMETERS")
    print("-" * 50)
    agent_cfg = data.get('agent_config', {})
    algo_params = agent_cfg.get('algorithm', {})
    print(f"Learning Rate:    {algo_params.get('learning_rate', 'N/A')}")
    print(f"Gamma (y):        {algo_params.get('gamma', 'N/A')}")
    print(f"Batch Size (Env): {agent_cfg.get('num_steps_per_env', 'N/A')}")

    # Section 4: Architecture
    if 'actor_cnn_cfg' in agent_cfg.get('policy', {}):
        print("\nCONVOLUTIONAL LAYER CONFIGURATION (ACTOR):")
        pprint.pprint(agent_cfg['policy']['actor_cnn_cfg'], indent=4)

    print("\n" + "="*80)
    print("="*80 + "\n")

if __name__ == "__main__":
    inspect_metadata()