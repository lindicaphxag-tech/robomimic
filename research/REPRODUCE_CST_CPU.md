# Independent CST/OSC reproduction — Panda CPU simulator

Clone the **public fork**, not a private research project:

```bash
git clone https://github.com/lindicaphxag-tech/robomimic.git
cd robomimic
git checkout validation/delta-actions-real-osc-20261008
bash research/reproduce_cst_cpu.sh
```

Environment: Ubuntu, Python 3.11, CPU, robosuite 1.5.2,
MuJoCo 3.3.0, headless OSMesa. On Ubuntu, install
`libosmesa6 libosmesa6-dev libgl1 libglx-mesa0` if needed.
No GPU, physical robot, training process or private dataset is required.

The script runs the actual official robosuite Panda OSC controller through:
- native action scale and SO(3) inverse;
- runtime nullspace posture-memory transfer and refusal;
- same-controller negative control (different vs identical MJCF);
- real simulator-generated HDF5 converter replay;
- Lift and Stack cross-task identical-MJCF delta->absolute control rollouts,
  three **action RNG seeds** each;
- online desired-goal controller memory tests, when validated.

The published canonical 15-test Lift+Stack run:
https://github.com/lindicaphxag-tech/robomimic/actions/runs/37713698317

The desired-goal extension is newly submitted and should only be cited as
validated after its own workflow succeeds.

## Report negative results

Please include your OS, Python versions, exact git commit, model versions,
exception traceback if any, and the per-trial printed `CST_MULTITASK_OSC`
/ `CST_REAL_DESIRED_STREAM` details. In particular, reproductions with
different controllers, gain schedules, interpolation memory or contact-rich
scenes that **fail** our checks are valuable counterexamples.

The tests are 8-step finite-horizon physical replay diagnostics. This is
*not* a trained frozen neural policy evaluation, hardware safety or universal
controller-equivalence certificate. No original public robomimic test suite
is claimed to have been run in full.
