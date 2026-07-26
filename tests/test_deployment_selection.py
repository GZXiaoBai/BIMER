from bimer.deployment_selection import select_deployment_model


def test_deployment_uses_v3_only_when_all_three_gates_pass():
    selection = {
        "state": "frozen",
        "version": "v3",
        "gate_ranking_weight": 0.05,
    }
    external = {"v3_acceptance": {"accepted": True}}
    m2 = {"passed": True}

    assert (
        select_deployment_model(
            frozen_selection=selection,
            external_report=external,
            m2_report=m2,
        )["deployed_model"]
        == "v3_ranked"
    )
    external["v3_acceptance"]["accepted"] = False
    fallback = select_deployment_model(
        frozen_selection=selection,
        external_report=external,
        m2_report=m2,
    )
    assert fallback["deployed_model"] == "v2_quality_lagf"
    assert fallback["fallback_used"] is True
