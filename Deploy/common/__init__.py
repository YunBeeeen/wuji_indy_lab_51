"""What every backend must reproduce identically.

The name says "shared", but sharing is the SYMPTOM, not the rule.  A module
belongs here when changing it invalidates a trained checkpoint -- when MuJoCo
and the physical hand disagreeing about it means the policy is being fed
something it never saw.  That covers three kinds of thing that look unrelated:

* Physical facts of the hand:  ``REAL_HAND_FACTORY_LIMITS``,
  ``DEPLOY_STIFFNESS_NM_PER_RAD`` / ``DAMPING`` / ``EFFORT_LIMITS``,
  ``CANONICAL_JOINTS``, the pregrasp pose, the fingertip frames.
* Pure convention with no physics in it:  the ``OBSERVATION_SLICES`` layout,
  ``ACTION_SCALE_RAD``, the OPEN/CLOSE one-hot, which branch a square stick's
  quaternion folds onto.  Isaac decided these; deployment does not get a vote.
* Interfaces:  ``backend_protocol.WujiBackend``,
  ``perception.StickPoseProvider``.  No values at all, just the shape both
  sides must agree on.

Being used in two places is NOT sufficient.  MuJoCo's joint storage order is a
real physical fact and lives in ``backends/joint_mapping.py``; the camera
extrinsics are measured and live in ``vision/deploy_rig.py``.  Neither exists on
the other side, so neither is something both sides must reproduce.

The practical test before adding a file here: *if this changed, would an
already-trained policy still be valid?*  If yes, it belongs somewhere else.

Nothing in here may import a simulator, a hand SDK, or a camera stack --
``policy/`` depends on this layer and has to load in all three environments.
``tests/test_common.py: PackageLayerTests`` enforces both halves of that.
"""
