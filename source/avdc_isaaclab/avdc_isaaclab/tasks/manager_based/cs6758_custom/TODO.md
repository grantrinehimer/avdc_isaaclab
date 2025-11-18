# cs6758_custom Adapter TODOs

This list tracks the gaps that must be closed before `MyPolicy_CL` can run
inside the IsaacLab version of the task via `IsaacLabEnvAdapter`.

1. **Camera sensors** – mount RGB + depth cameras that mimic the MetaWorld
   `corner/corner2/corner3` viewpoints and expose them through Isaac's sensor
   interface.
2. **Projection matrices** – export per-camera 3×4 projection matrices (or
   intrinsic/extrinsic parameters) so `fetch_intrinsics()` can reproduce the
   `get_cmat` output expected by the rigid-transform solver.
3. **Segmentation IDs** – define persistent instance IDs for manipulated objects
   and map them to the `name2maskid` keys used by AVDC experiments.
4. **Sensor readout path** – add a lightweight data pipe (render hooks or custom
   sensor readers) that returns RGB-D frames every time the adapter calls
   `fetch_rgbd`, without blocking the simulator.
5. **Physical sensor assets** – add the missing task sensors (mentioned in the
   user notes) and ensure the reset events keep object/basket poses aligned with
   the planning assumptions.

Once these steps are complete, the stubs in `env_interfaces/isaaclab.py` can be
implemented and the Isaac benchmark script will run end-to-end.

