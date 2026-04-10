# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

def list_all_gymnasium_environments():
    """List all environments registered in gymnasium."""
    try:
        import gymnasium as gym
        print(f"\nAll environments registered in Gymnasium:")
        env_ids = list(gym.envs.registry.keys())
        for env_id in sorted(env_ids):
            print(f"  ✓ {env_id}")
        print(f"Total gymnasium environments: {len(env_ids)}")
    except ImportError:
        print("Gymnasium not available for listing environments.")
    except Exception as e:
        print(f"Error listing gymnasium environments: {e}")