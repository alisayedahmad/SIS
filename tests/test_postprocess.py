"""Tests pour le post-traitement des masques de segmentation."""

import numpy as np
import pytest
from src.postprocess import postprocess_prediction


def make_mask(h=64, w=64, fill=True):
    """Masque synthétique avec quelques blobs de bâtiments."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if fill:
        mask[10:30, 10:30] = 1  # gros bloc
        mask[40:45, 40:45] = 1  # petit bloc
        mask[5, 5] = 1          # pixel isolé (bruit)
    return mask


class TestBuildings:

    def test_output_shape(self):
        mask = make_mask()
        result = postprocess_prediction(mask[None], kind="buildings")
        assert result.shape == (1, 64, 64)

    def test_removes_isolated_pixels(self):
        # Un pixel isolé doit disparaître après le nettoyage
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[32, 32] = 1
        result = postprocess_prediction(mask[None], kind="buildings", min_size=10)
        assert result.sum() == 0

    def test_keeps_large_blobs(self):
        # Un grand bloc doit survivre au post-traitement
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:40, 10:40] = 1
        result = postprocess_prediction(mask[None], kind="buildings", min_size=10)
        assert result.sum() > 0

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = postprocess_prediction(mask[None], kind="buildings")
        assert result.sum() == 0
        assert result.shape == (1, 64, 64)

    def test_2d_input(self):
        # Doit accepter (H,W) en plus de (1,H,W)
        mask = make_mask()
        result = postprocess_prediction(mask, kind="buildings")
        assert result.shape == (1, 64, 64)


class TestRoads:

    def test_output_shape(self):
        mask = make_mask()
        result = postprocess_prediction(mask[None], kind="roads")
        assert result.shape == (1, 64, 64)

    def test_skeletonization_reduces_width(self):
        # Une bande de 8 pixels de haut doit être réduite à ~1-3 pixels après squelettisation
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[28:36, 5:60] = 1  # bande horizontale de 8 pixels de haut
        result = postprocess_prediction(mask[None], kind="roads")
        # On vérifie l'épaisseur sur l'axe vertical (nombre de lignes non vides)
        rows_with_pixels = (result[0].sum(axis=1) > 0).sum()
        assert rows_with_pixels < 8

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = postprocess_prediction(mask[None], kind="roads")
        assert result.sum() == 0


class TestMulti:

    def test_output_shape(self):
        mask = make_mask()
        result = postprocess_prediction(mask[None], kind="multi")
        assert result.shape == (1, 64, 64)

    def test_does_not_crash(self):
        mask = make_mask()
        postprocess_prediction(mask[None], kind="multi")

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        result = postprocess_prediction(mask[None], kind="multi")
        assert result.sum() == 0


def test_unknown_kind_does_not_crash():
    # Un type inconnu doit juste logger un warning, pas crasher
    mask = make_mask()
    result = postprocess_prediction(mask[None], kind="unknown_type")
    assert result.shape == (1, 64, 64)