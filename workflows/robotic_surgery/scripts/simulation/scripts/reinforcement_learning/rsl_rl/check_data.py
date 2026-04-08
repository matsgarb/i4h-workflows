import numpy as np
import os
import pprint
import json

NPY_PATH = "/home/dvrkteam/i4h-workflows/workflows/robotic_surgery/scripts/simulation/scripts/reinforcement_learning/rsl_rl/logs/rsl_rl/psm_reach/2026-03-13_10-03-18/env_agent_data.npy"

def inspect_metadata():
    if not os.path.exists(NPY_PATH):
        print(f"ERROR: Target file not found at:\n{NPY_PATH}")
        return

    try:
        data = np.load(NPY_PATH, allow_pickle=True).item()
    except Exception as e:
        print(f"ERROR: Failed to load NPY file: {e}")
        return

    print("\n" + "="*80)
    print("TRAINING DATA INSPECTION REPORT")
    print("="*80)

    # Display the structure of the data
    print("\nDATA STRUCTURE (Top-level keys):")
    print("-" * 50)
    if isinstance(data, dict):
        for key in data.keys():
            value = data[key]
            if isinstance(value, dict):
                print(f"{key:<20} : Dictionary with keys: {list(value.keys())[:5]}")
            elif isinstance(value, (list, np.ndarray)):
                print(f"{key:<20} : Array/List (length: {len(value)})")
            else:
                print(f"{key:<20} : {type(value).__name__}")
    else:
        print(f"Data type: {type(data)}")
        print(f"Data content:\n{data}")

    # Section 1: Observation Information
    print("\nSECTION 1: OBSERVATION SPECIFICATIONS")
    print("-" * 50)
    obs_info = data.get('data', {}).get('obs_info', {})
    if obs_info:
        print(f"{'Key':<15} | {'Shape':<22} | {'Dtype':<15}")
        print("-" * 50)
        for key, info in obs_info.items():
            print(f"{key:<15} | {str(info['shape']):<22} | {info['dtype']}")
    else:
        print("No observation info found in data structure")

    # Section 2: Agent and Environment Summary
    print("\nSECTION 2: AGENT AND ENVIRONMENT SUMMARY")
    print("-" * 50)
    main_data = data.get('data', {})
    print(f"Algorithm Class:  {main_data.get('algo_class', 'N/A')}")
    print(f"Policy Class:     {main_data.get('policy_class', 'N/A')}")
    print(f"Num Environments: {main_data.get('num_envs', 'N/A')}")
    print(f"Action Dimension: {main_data.get('num_actions', 'N/A')}")

    # Section 3: Training Hyperparameters
    print("\nSECTION 3: TRAINING HYPERPARAMETERS")
    print("-" * 50)
    agent_cfg = data.get('agent_config', {})
    if agent_cfg:
        algo_params = agent_cfg.get('algorithm', {})
        print(f"Learning Rate:    {algo_params.get('learning_rate', 'N/A')}")
        print(f"Gamma (Discount): {algo_params.get('gamma', 'N/A')}")
        print(f"Batch Size:       {agent_cfg.get('num_steps_per_env', 'N/A')}")
    else:
        print("No agent configuration found")

    # Section 4: Network Architecture
    print("\nSECTION 4: NETWORK ARCHITECTURE")
    print("-" * 50)
    if agent_cfg and 'policy' in agent_cfg:
        policy = agent_cfg['policy']
        if 'actor_cnn_cfg' in policy:
            print("CNN Configuration (Actor):")
            pprint.pprint(policy['actor_cnn_cfg'], indent=4)
        if 'actor_hidden_dims' in policy:
            print(f"Actor Hidden Dimensions: {policy.get('actor_hidden_dims', 'N/A')}")
        if 'critic_hidden_dims' in policy:
            print(f"Critic Hidden Dimensions: {policy.get('critic_hidden_dims', 'N/A')}")
    else:
        print("No policy/network configuration found")

    print("\n" + "="*80)
    print("="*80 + "\n")

if __name__ == "__main__":
    inspect_metadata()