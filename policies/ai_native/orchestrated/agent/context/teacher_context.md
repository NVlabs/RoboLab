# Project and teacher context

## Baseline and what is known

The student baseline is the remote Cosmos3 action policy, not retired RLinf/PyTorch or
VoLo grasp/place macros. Student inference proposes H{{student_action_horizon}} absolute joint targets;
the teacher reviews each new proposal using an outcome/intent gate and can
redirect using short no-rollback EEF segments. Native joint position control
continues underneath both sources. This rollout does not train the student.

The colleague's ordered-block run (`generic_gate_ordered_blocks_20260909_01`,
executing commit `931b194`, reported in commit `5ca5512`) reached native success
at 557 controls: 482 student, 5 edited, 70 absolute EEF; 48 decisions and 15
interventions. Order was red, blue, green, yellow from bottom to top. The teacher
was the existing conversation assistant with prior task-code knowledge, not a
fresh blind reviewer. Its complete earlier conversation and exact model/effort
are not available here; this document does not pretend to reproduce them.

The report describes incorrect student object selection (yellow before blue
or green), release before reaching the required destination, short redirections
with explicit gripper retention, then handing back for transport/alignment.
One correction itself went in the wrong direction and was revised using the
next observation. Success fired on the first final yellow-release control;
four-layer settling after withdrawal was not tested. These are examples of
reasoning and failure modes, NOT action coordinates or a script for a new run.

Our isolated-agent run `ordered_blocks_codex_tools_20260910_06` was stopped by
the user at 735 controls, 555 student + 180 EEF, 73 completed decisions. Its
decision records reported green-layer loss and rebuilding the lower layers.
It was incomplete, not a native failed or successful episode. In that harness
only the three service tools were enabled. Tool availability is specified by the current rollout configuration. Do not assume the new initial layout matches either run.

## Useful operating knowledge

- Derive prerequisites from the instruction returned by the environment. For
  ordered stacking the red base may already rest on the table: moving every
  block is not inherently necessary. Verify current support visually.
- The student always receives the original instruction; temporary teacher
  subgoals belong in this conversation, not a rewritten student prompt.
- FK predicts robot target poses only. It does not predict contact, object
  attachment, collision, or successful placement. A commanded closed gripper
  is not evidence of a grasp; check object motion with measured robot motion.
- A target may not be reached during a short correction. Compare actual
  pre/post EEF poses with the requested target and review IK diagnostics when
  useful. Retain uncertainty rather than marking an intended motion complete.
- Use all attached camera views and earlier observations in this conversation.
  Do not use private object truth, rewards, or future simulation to plan actions.
  No hidden grasp/place planner exists.
