import numpy as np
import pandas as pd

import autoeis as ae


def test_assess_posterior_sample_classifies_normal_distribution():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=10, scale=2, size=2000)

    result = ae.diagnostics.assess_posterior_sample(sample, "R1", threshold=75)

    assert result.parameter == "R1"
    assert result.shape == "Normal"
    assert result.status == "Good"
    assert result.r2 > 0.99
    assert result.modes == 1


def test_assess_posterior_sample_detects_multipeak_distribution():
    rng = np.random.default_rng(1)
    sample = np.concatenate(
        [
            rng.normal(loc=-3, scale=0.25, size=1000),
            rng.normal(loc=3, scale=0.25, size=1000),
        ]
    )

    result = ae.diagnostics.assess_posterior_sample(sample, "P2w", threshold=75)

    assert result.shape == "Multi-Peak"
    assert result.status == "Bad"
    assert result.modes == 2


def test_assess_posterior_quality_accepts_sample_mapping_and_skips_auxiliary():
    rng = np.random.default_rng(2)
    samples = {
        "R1": rng.normal(size=500),
        "P2n": rng.lognormal(mean=0, sigma=0.8, size=500),
        "lp__": rng.normal(size=500),
    }

    diagnostics = ae.diagnostics.assess_posterior_quality(samples, threshold=75)

    assert list(diagnostics["parameter"]) == ["R1", "P2n"]
    assert "lp__" not in diagnostics["parameter"].to_list()
    assert set(
        [
            "parameter",
            "score",
            "r2",
            "skewness",
            "modes",
            "shape",
            "status",
            "good",
        ]
    ).issubset(diagnostics.columns)


def test_plot_posterior_quality_saves_png(tmp_path):
    rng = np.random.default_rng(3)
    sample = rng.normal(size=500)

    fig, _ = ae.diagnostics.plot_posterior_quality(
        sample, parameter="sigma.mag", save_dir=tmp_path
    )

    try:
        saved = list(tmp_path.glob("sigma.mag-*.png"))
        assert len(saved) == 1
    finally:
        fig.clf()


def test_add_posterior_quality_updates_and_ranks_circuits_dataframe():
    class FakeInferenceResult:
        def __init__(self, circuit, samples):
            self.circuit = circuit
            self.samples = samples
            self.converged = True

    rng = np.random.default_rng(4)
    results = [
        FakeInferenceResult("R1", {"R1": rng.normal(size=1000)}),
        FakeInferenceResult(
            "R1-[R2,P3]",
            {
                "R1": np.concatenate(
                    [
                        rng.normal(loc=-3, scale=0.2, size=500),
                        rng.normal(loc=3, scale=0.2, size=500),
                    ]
                )
            },
        ),
    ]
    circuits = pd.DataFrame({"circuitstring": ["R1", "R1-[R2,P3]"]})

    ranked = ae.diagnostics.add_posterior_quality(circuits, results, threshold=75)

    assert "Posterior Score" in ranked
    assert "Posterior Quality" in ranked
    assert list(ranked["Posterior Score Rank"]) == [1, 2]
    assert ranked.iloc[0]["Bad Parameters"] == "None"
