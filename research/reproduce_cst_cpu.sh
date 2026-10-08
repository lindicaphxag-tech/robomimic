#!/usr/bin/env bash
# Standalone no-GPU reproduction of the CST real OSC migration results.
# Run from the root of the public lindicaphxag-tech/robomimic fork.
# Ubuntu packages required (if missing):
#   sudo apt-get install libosmesa6 libosmesa6-dev libgl1 libglx-mesa0
# Original and negative-control failures are preserved by pytest.
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m pip install "robosuite==1.5.2" "mujoco==3.3.0" numpy scipy pytest h5py
python - <<'PY'
import mujoco, robosuite
print("CST_ENVIRONMENT", {"mujoco":mujoco.__version__, "robosuite":robosuite.__version__})
PY
python -m pytest -q -s \
  tests/test_robosuite_action_conversion.py \
  tests/test_robosuite_real_osc_inverse.py \
  tests/test_robosuite_paired_closed_loop.py \
  tests/test_robosuite_hdf5_converter_integration.py \
  tests/test_robosuite_simulator_negative_control.py \
  tests/test_cst_real_osc_handshake.py \
  tests/test_cst_real_osc_multitask.py \
  tests/test_cst_real_desired_osc.py
