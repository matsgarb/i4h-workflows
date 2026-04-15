# Setup

## System Requirements
- OS: Ubuntu 22.04
- Python: 3.10.19
- Isaac Sim installed at `~/isaacsim/`

## Environment activation
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robotic_surgery
source ~/isaacsim/setup_conda_env.sh
export PYTHONPATH="/home/dvrkteam/i4h-workflows/workflows/robotic_surgery/scripts:$PYTHONPATH"
```

## Install dependencies
```bash
pip install -r requirements.txt
```