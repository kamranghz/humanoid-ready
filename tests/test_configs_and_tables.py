"""Tracked configs parse, the pre-registered decision rule is complete, joint tables are consistent."""

from pathlib import Path

import yaml

from hready.body.joint_indices import (
    FEET_JOINTS,
    FOOT_CONTACT_CHANNEL_JOINTS,
    JOINT_NAMES_22,
    LOWER_BODY_JOINTS,
    PELVIS,
    UPPER_BODY_JOINTS,
)

ROOT = Path(__file__).resolve().parents[1]
TRACKED_CONFIGS = [p for p in sorted((ROOT / "configs").glob("*.yaml")) if p.name != "paths.yaml"]


def test_all_tracked_configs_parse():
    assert TRACKED_CONFIGS
    for p in TRACKED_CONFIGS:
        assert isinstance(yaml.safe_load(p.read_text(encoding="utf-8")), dict), p.name


def test_preregistered_decision_rule_lists_all_gates():
    cfg = yaml.safe_load((ROOT / "configs" / "evidence_locked_preregistered_comparison.yaml").read_text(encoding="utf-8"))
    rule = cfg["decision_rule"]
    assert set(rule["gates"]) == {
        "mpjpe_full_hidden_mm",
        "mpjpe_full_all_mm",
        "penetration_mean_mm",
        "foot_skate_m_s",
        "contact_ece",
    }
    assert set(rule["candidates"]) == set(cfg["arms"])
    assert cfg["anchor"]["eligible"] is False


def test_citation_metadata_present():
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["license"] == "MIT" and cff["authors"]


def test_joint_tables_are_consistent():
    assert len(JOINT_NAMES_22) == 22
    assert FOOT_CONTACT_CHANNEL_JOINTS == (7, 10, 8, 11)
    assert set(FOOT_CONTACT_CHANNEL_JOINTS) == set(FEET_JOINTS)
    assert not set(UPPER_BODY_JOINTS) & set(LOWER_BODY_JOINTS)
    assert set(UPPER_BODY_JOINTS) | set(LOWER_BODY_JOINTS) | {PELVIS} == set(range(22))
