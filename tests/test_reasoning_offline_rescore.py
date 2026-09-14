from scripts.rescore_reasoning_outputs_offline import (
    build_raw_json_parse_audit_from_conditions,
    rebuild_parsed,
)


def test_rebuild_parsed_detects_layer_steps_without_pipeline_type() -> None:
    parsed, source = rebuild_parsed(
        {
            "steps": {
                "L1": {
                    "raw": '{"coordinates": "(0,0), (1,1)", "intercepts": {"x_intercepts": [0], "y_intercept": 0}, "extrema": []}'
                },
                "L2": {
                    "raw": '{"predictions": {"0.5": 0.5}, "rate_of_change": 1, "monotonicity": [{"interval": [-inf, +inf], "direction": "increasing"}], "symmetry": "odd"}'
                },
                "L3": {
                    "raw": '{"expression": "x", "domain": "(-inf, +inf)", "range": "(-inf, +inf)", "inference_explanation": "A line through the origin."}'
                },
            },
            "final_parsed": {"expression": "stale"},
        }
    )

    assert source == "layer_raw_reparse"
    assert parsed["expression"] == "x"
    assert parsed["domain"] == "(-inf, +inf)"
    assert parsed["monotonicity"] == [
        {"interval": [float("-inf"), float("inf")], "direction": "increasing"}
    ]


def test_raw_json_parse_audit_counts_layer_raw_outputs() -> None:
    audit = build_raw_json_parse_audit_from_conditions(
        [
            {
                "sample_id": "s0",
                "condition": "E5",
                "condition_record": {
                    "pipeline": {
                        "steps": {
                            "L1": {"raw": '{"coordinates": "(0,0)"}'},
                            "L2": {"raw": '{"monotonicity": [{"interval": [-inf, 0], "direction": "increasing"}]}'},
                            "L3": {"raw": '{"expression": "x", "comments": "as x \\to \\infty"}'},
                        }
                    }
                },
            }
        ]
    )

    e5 = audit["by_condition"]["E5"]
    assert e5["raw_outputs"] == 3
    assert e5["bare_infinity_outputs"] == 1
    assert e5["strict_json_fail_outputs"] == 2
    assert audit["by_condition_step"]["E5"]["L2"]["strict_json_fail_outputs"] == 1
    assert audit["by_condition_step"]["E5"]["L3"]["strict_json_fail_outputs"] == 1
