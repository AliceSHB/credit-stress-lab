"""Sanity tests: the chain has to hold end to end."""
import numpy as np
import pandas as pd
import pytest

from credit_stress.data import load_loans, load_macro
from credit_stress.model import calibrate_level
from credit_stress.satellite import fit_satellite, scenario_multipliers
from credit_stress.limits import assess, build_book, build_register, reconcile
from credit_stress.stress import run_stress


@pytest.fixture(scope="module")
def loans():
    return load_loans()


@pytest.fixture(scope="module")
def macro():
    return load_macro()


def test_loan_file_has_target_and_exposure(loans):
    assert loans["DEFAULT"].isin([0, 1]).all()
    assert (loans["EAD"] >= 0).all()
    assert len(loans) > 20000


def test_calibration_hits_the_target():
    rng = np.random.default_rng(0)
    p = rng.beta(2, 20, 5000)
    out, _ = calibrate_level(p, 0.035)
    assert abs(out.mean() - 0.035) < 1e-4
    assert ((out >= 0) & (out <= 1)).all()


def test_scenarios_are_ordered(macro):
    fit = fit_satellite(macro)
    mult = scenario_multipliers(fit, macro).set_index("scenario")["pd_multiplier"]
    assert mult["baseline"] == pytest.approx(1.0)
    assert mult["severe"] > mult["adverse"] > mult["baseline"]
    assert mult["severe"] <= 3.0


def test_stress_is_monotone_and_additive(loans, macro):
    fit = fit_satellite(macro)
    mult = scenario_multipliers(fit, macro)
    pd12 = np.full(len(loans), 0.035)
    res_df, trig = run_stress(loans, pd12, mult)
    res = res_df.set_index("scenario")

    assert res.loc["severe", "ecl"] > res.loc["adverse", "ecl"] > res.loc["baseline", "ecl"]
    assert res.loc["baseline", "ecl_delta"] == pytest.approx(0.0, abs=1e-9)
    for s in res.index:
        assert res.loc[s, "coverage"] == pytest.approx(res.loc[s, "ecl"] / res.loc[s, "ead"])
    assert set(trig.columns) == {"trigger", "threshold", "observed", "fired"}


def test_limit_cuts_reconcile_or_are_declared_partial():
    book = build_book()
    register = build_register(book)
    rec = reconcile(book, register)

    assert rec["full_cuts_reconcile"], "sub-portfolio, sector and country must sum to total exposure"
    assert 0 < rec["obligor_coverage"] <= 1
    reg = assess(register)
    assert set(reg["state"]).issubset({"Within appetite", "Early warning", "Breach"})
    assert (reg["headroom"] == reg["limit"] - reg["exposure"]).all()
