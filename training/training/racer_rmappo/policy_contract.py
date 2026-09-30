"""Versioned actor/backend contracts; no simulator imports."""
NAVIGATION = "navigation_v2"
LEGACY = "hybrid_v1"


def policy_version(cfg):
    version = cfg.get("policy", {}).get("version", NAVIGATION)
    if version not in (NAVIGATION, LEGACY):
        raise ValueError(f"unknown policy version: {version}")
    return version


def policy_spec(version):
    if version not in (NAVIGATION, LEGACY):
        raise ValueError(f"unknown policy version: {version}")
    keys = ["depth", "ego", "target", "neighbors"]
    if version == LEGACY:
        keys += ["candidates", "decision_mask"]
    return {"version": version, "observation_keys": keys,
            "action_dim": 4 if version == NAVIGATION else 9}


def checkpoint_policy(checkpoint):
    # Only old files without a spec may infer the historical hybrid layout.
    spec = checkpoint.get("policy_spec")
    if spec is None:
        if "candidate_query.weight" not in checkpoint["actor"]:
            raise ValueError("unversioned checkpoint has no recognized legacy actor")
        return LEGACY
    version = spec.get("version")
    if spec != policy_spec(version):
        raise ValueError("checkpoint policy contract is inconsistent")
    return version
