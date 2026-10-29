import pytest

from research0 import stats


def test_wilson_interval():
    p, low, high = stats.wilson(0, 100)
    assert p == 0 and low == 0 and 0.03 < high < 0.04      # 0 of 100 does not mean "zero rate": the upper bound is about 3.7%
    assert stats.wilson(5, 0) == (None, None, None)
    p, low, high = stats.wilson(50, 100)
    assert abs(p - 0.5) < 1e-9 and 0.40 < low < 0.41 and 0.59 < high < 0.60


def test_kappa_known_values():
    perfect = stats.cohen_kappa([("a", "a"), ("b", "b"), ("c", "c"), ("a", "a")])
    assert perfect["kappa"] == 1 and perfect["agreement"] == 1
    # By hand: 20 of 25 agree (0.8); marginals A 18/7, B 17/8 -> chance (18*17+7*8)/625 = 0.5792 -> kappa = 0.2208/0.4208
    assert stats.cohen_kappa([("y", "y")] * 15 + [("n", "n")] * 5 + [("y", "n")] * 3 + [("n", "y")] * 2)["kappa"] == pytest.approx(0.5247, abs=1e-4)
    assert stats.cohen_kappa([("y", "y")] * 5)["kappa"] is None   # one category only: chance agreement 1, kappa undefined, not 1
    with pytest.raises(ValueError):
        stats.cohen_kappa([])
    three = stats.cohen_kappa([("yes", "yes"), ("no", "uncertain"), ("no", "no"), ("uncertain", "uncertain")])
    assert 0 < three["kappa"] < 1


def test_krippendorff_alpha():
    assert stats.krippendorff_alpha_nominal([["a", "a"], ["b", "b"], ["a", "a"], ["b", "b"]])["alpha"] == pytest.approx(1)
    assert stats.krippendorff_alpha_nominal([["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]])["alpha"] < 0
    # Known example from Krippendorff (2011) shape: with missing ratings still computable
    value = stats.krippendorff_alpha_nominal([["a", "a", "a"], ["a", "b"], ["b", "b"], ["b"]])
    assert value["units"] == 3 and 0 < value["alpha"] < 1
    with pytest.raises(ValueError):
        stats.krippendorff_alpha_nominal([["a"]])


def test_prf_and_macro_f1():
    assert stats.prf(8, 2, 2) == {"precision": 0.8, "recall": 0.8, "f1": 0.8, "tp": 8, "fp": 2, "fn": 2}
    assert stats.prf(0, 0, 0)["f1"] is None
    truth, pred = ["y", "y", "n", "n", "n", "n"], ["y", "n", "n", "n", "n", "y"]
    assert stats.macro_f1(truth, pred, ["y", "n"]) == pytest.approx((0.5 + 0.75) / 2)


def test_icc_and_spearman():
    perfect = [[1, 1], [2, 2], [3, 3], [4, 4]]
    assert stats.icc_agreement(perfect)["icc"] == pytest.approx(1)
    shifted = [[1, 2], [2, 3], [3, 4], [4, 5]]     # same ordering, systematic offset: absolute agreement is penalised
    assert stats.icc_agreement(shifted)["icc"] < 0.9
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40])["rho"] == pytest.approx(1)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1])["rho"] == pytest.approx(-1)
    with pytest.raises(ValueError):
        stats.icc_agreement([[1, 2]])
