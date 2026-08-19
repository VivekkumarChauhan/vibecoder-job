"""tests/test_stage1_camera.py — Unit tests for Stage 1: Camera Discovery."""
from __future__ import annotations

import pytest
from pipeline.schemas import CameraRole


@pytest.mark.unit
def test_camera_inventory_get_by_role(three_camera_inventory):
    host_cam = three_camera_inventory.get_camera_by_role(CameraRole.HOST_HERO)
    assert host_cam is not None
    assert host_cam.camera_id == "cam_1"

    wide_cam = three_camera_inventory.get_camera_by_role(CameraRole.WIDE)
    assert wide_cam is not None
    assert wide_cam.camera_id == "cam_3"

    guest_cam = three_camera_inventory.get_camera_by_role(CameraRole.GUEST_HERO)
    assert guest_cam is not None
    assert guest_cam.camera_id == "cam_2"


@pytest.mark.unit
def test_camera_inventory_get_valid_cameras(frozen_camera_inventory):
    valid = frozen_camera_inventory.get_valid_cameras()
    assert len(valid) == 1
    assert valid[0].camera_id == "cam_2"


@pytest.mark.unit
def test_single_camera_inventory_valid(single_camera_inventory):
    assert single_camera_inventory.total_cameras == 1
    assert len(single_camera_inventory.get_valid_cameras()) == 1


@pytest.mark.unit
def test_empty_inventory_no_valid_cameras(empty_inventory):
    assert len(empty_inventory.get_valid_cameras()) == 0
    assert empty_inventory.total_cameras == 0


@pytest.mark.unit
def test_camera_role_not_found(three_camera_inventory):
    result = three_camera_inventory.get_camera_by_role(CameraRole.SECONDARY)
    assert result is None


@pytest.mark.unit
def test_frozen_camera_not_in_valid(frozen_camera_inventory):
    valid_ids = {c.camera_id for c in frozen_camera_inventory.get_valid_cameras()}
    assert "cam_1" not in valid_ids, "Frozen cam_1 should not be in valid cameras"
    assert "cam_2" in valid_ids
