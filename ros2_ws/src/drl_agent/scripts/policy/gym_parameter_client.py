"""ROS2 parameter client for the curriculum environment node (``/gym_node``).

Extracted from ``TrainTQCCurriculum`` so all ``/gym_node`` parameter get/set
interaction lives behind one cohesive object instead of being scattered across
the trainer as a handful of ``_set_*`` / ``_fetch_*`` methods that each open-code
the same request/response plumbing.

``GymParameterClient`` owns the two ``rcl_interfaces`` service clients and uses
the trainer node's robust ``_call_service`` helper (bounded waits + retries +
fail-fast ``EnvServiceError``). Transport/service failures propagate as
``EnvServiceError`` exactly as before, so the trainer's checkpoint-on-failure
path is unchanged.

The trainer keeps thin ``_set_curriculum_stage`` / ``_fetch_*`` wrappers that
delegate here and apply any trainer-local state updates (e.g. caching the new
stage), preserving every existing call site and behaviour.
"""

from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

from environment_interface import EnvServiceError


class GymParameterClient:
    """Encapsulates get/set parameter calls against the ``/gym_node`` node.

    Parameters
    ----------
    node:
        The trainer ROS2 node. Must provide ``create_client``, ``get_logger``
        and the ``_call_service`` helper (from ``EnvironmentInterface``).
    """

    def __init__(self, node):
        self._node = node
        # Node is named "gym_node" — matches Environment.__init__("gym_node").
        self._set_client = node.create_client(
            SetParameters, "/gym_node/set_parameters"
        )
        self._get_client = node.create_client(
            GetParameters, "/gym_node/get_parameters"
        )

    # -- back-compat accessors (the trainer used to hold these directly) ------
    @property
    def set_client(self):
        return self._set_client

    @property
    def get_client(self):
        return self._get_client

    def _log(self):
        return self._node.get_logger()

    def set_curriculum_stage(self, stage: int) -> bool:
        """Push curriculum_stage to /gym_node via set_parameters service.

        Transport/service failures raise EnvServiceError so training can
        checkpoint and stop instead of continuing with an unknown stage.
        Returns True when the env accepted the new stage."""
        req = SetParameters.Request()
        req.parameters = [
            Parameter(
                name="curriculum_stage",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER,
                    integer_value=int(stage),
                ),
            )
        ]
        response = self._node._call_service(
            self._set_client,
            req,
            "gym_node/set_parameters",
            wait_timeout=3.0,
            call_timeout=5.0,
        )
        ok = bool(response.results) and all(r.successful for r in response.results)
        if ok:
            self._log().info(
                f"[Curriculum] Environment stage set to {stage}."
            )
        else:
            self._log().warn(
                f"[Curriculum] set_parameters for stage={stage} rejected by gym_node."
            )
        return ok

    def get_eval_mode(self):
        """Read the env's ACTUAL curriculum_eval_mode bool via get_parameters.
        Returns True/False, or None when the param is absent on an older env.
        Transport/service failures raise EnvServiceError so the trainer can
        fail-fast and checkpoint instead of silently proceeding with a broken
        control path."""
        req = GetParameters.Request()
        req.names = ["curriculum_eval_mode"]
        res = self._node._call_service(
            self._get_client,
            req,
            "gym_node/get_parameters",
            wait_timeout=1.0,
            call_timeout=3.0,
        )
        if not res.values:
            return None
        pv = res.values[0]
        if pv.type == ParameterType.PARAMETER_BOOL:
            return bool(pv.bool_value)
        return None

    def set_eval_mode(self, on: bool) -> bool:
        """Set the env's curriculum_eval_mode so evaluation episodes use
        eval_map_types (round-robin) instead of the training map distribution,
        while the env stays in train mode (obstacles/humans still activate).

        Returns True only after CONFIRMING via get_parameters that the env's
        actual value equals `on` — not merely that the SetParameters request was
        accepted. This makes the caller's eval_map_applied flag reflect ground
        truth even when a set times out but applies, or the env was already in the
        target state from a prior (failed-looking) toggle.
        """
        req = SetParameters.Request()
        req.parameters = [
            Parameter(
                name="curriculum_eval_mode",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=bool(on),
                ),
            )
        ]
        res = self._node._call_service(
            self._set_client,
            req,
            "gym_node/set_parameters",
            wait_timeout=3.0,
            call_timeout=5.0,
        )
        if not res.results or not all(r.successful for r in res.results):
            self._log().warn(
                f"[Curriculum] set_parameters for eval_mode={on} rejected by gym_node."
            )
        # Confirm the actually-applied value (ground truth), regardless of the
        # request-level result above.
        actual = self.get_eval_mode()
        return actual is not None and actual == bool(on)

    def get_num_stages(self) -> int:
        """Query curriculum_num_stages from the running gym_node.

        EnvironmentCurriculum declares this parameter at startup with the exact
        count from the config file the environment was actually launched with,
        so trainer and environment always agree on stage count.
        Falls back to 5 only when the parameter is absent/invalid; actual
        transport/service failures raise EnvServiceError so the run fails-fast.
        """
        req = GetParameters.Request()
        req.names = ["curriculum_num_stages"]
        response = self._node._call_service(
            self._get_client,
            req,
            "gym_node/get_parameters",
            wait_timeout=3.0,
            call_timeout=5.0,
        )
        if not response.values:
            self._log().warn(
                "[Curriculum] curriculum_num_stages not found on gym_node — "
                "defaulting to 5."
            )
            return 5
        n = int(response.values[0].integer_value)
        if n < 1:
            return 5
        self._log().info(f"[Curriculum] gym_node reports {n} stages.")
        return n

    def get_env_aux_params(self) -> dict:
        """AUX_ABLATION: read the env node's actual loaded-config path / hash and
        running aux geometry from gym_node, so run_manifest.json records the TRUE
        env settings instead of a re-discovered file.  Returns {} on any failure.
        """
        out = {}
        try:
            names = ["loaded_config_path", "loaded_config_sha1", "aux_enabled",
                     "aux_num_sectors", "aux_horizons_sec", "aux_risk_distance_scale"]
            req = GetParameters.Request()
            req.names = names
            res = self._node._call_service(
                self._get_client,
                req,
                "gym_node/get_parameters",
                wait_timeout=3.0,
                call_timeout=5.0,
            )
            if not res.values:
                return out
            for name, pv in zip(names, res.values):
                t = pv.type
                if t == ParameterType.PARAMETER_STRING:
                    out[name] = pv.string_value
                elif t == ParameterType.PARAMETER_BOOL:
                    out[name] = bool(pv.bool_value)
                elif t == ParameterType.PARAMETER_INTEGER:
                    out[name] = int(pv.integer_value)
                elif t == ParameterType.PARAMETER_DOUBLE:
                    out[name] = float(pv.double_value)
                elif t == ParameterType.PARAMETER_DOUBLE_ARRAY:
                    out[name] = [float(x) for x in pv.double_array_value]
                # PARAMETER_NOT_SET -> param absent on this env (older build)
        except Exception as e:
            self._log().warn(f"[AUX_ABLATION] could not read env aux params: {e}")
        return out

    def get_current_map_type(self) -> str:
        """Read the env's read-only `current_map_type` parameter (structured map
        curriculum). Returns "" when unavailable (e.g. layout disabled / older
        env). Cheap: one get_parameters round-trip, called once per episode."""
        try:
            req = GetParameters.Request()
            req.names = ["current_map_type"]
            res = self._node._call_service(
                self._get_client,
                req,
                "gym_node/get_parameters",
                wait_timeout=1.0,
                call_timeout=3.0,
            )
            if not res.values:
                return ""
            pv = res.values[0]
            if pv.type == ParameterType.PARAMETER_STRING:
                return pv.string_value
        except EnvServiceError:
            return ""
        return ""
