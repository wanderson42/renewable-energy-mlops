"""Print an allowlisted review; provider metadata and input values remain private."""

import json
import re
import sys


def selected_settings(resource):
    return {item["name"]: item["value"] for item in (resource or {}).get("set", [])
            if item["name"] in ("api.enabled", "repro.chartHash")}


def safe_value(field, value):
    if value is None:
        return None
    patterns = {"api.enabled": r"true|false", "repro.chartHash": r"[a-f0-9]{64}",
                "version": r"\d+\.\d+\.\d+"}
    return value if isinstance(value, str) and re.fullmatch(patterns[field], value) else "[redacted/unexpected]"


def review(plan):
    if plan.get("errored"):
        raise RuntimeError("O plano informa erro; inspecione o log privado.")
    counts = {"add": 0, "change": 0, "destroy": 0}
    resources = []
    for item in plan.get("resource_changes", []):
        change = item["change"]
        actions = change["actions"]
        counts["add"] += int("create" in actions)
        counts["change"] += int("update" in actions)
        counts["destroy"] += int("delete" in actions)
        if item["address"] != "helm_release.mlops":
            resources.append({"resource": "[unexpected resource]"})
            continue
        before, after = change.get("before") or {}, change.get("after") or {}
        settings_before, settings_after = selected_settings(before), selected_settings(after)
        resource = {"resource": "helm_release.mlops", "actions": actions,
                    "release_is_rehearsal": after.get("name") == "energy-mlops-repro",
                    "namespace_is_rehearsal": after.get("namespace") == "energy-mlops-repro",
                    "chart_version": {"before": safe_value("version", before.get("version")),
                                      "after": safe_value("version", after.get("version"))},
                    "settings": {field: {"before": safe_value(field, settings_before.get(field)),
                                         "after": safe_value(field, settings_after.get(field))}
                                 for field in ("api.enabled", "repro.chartHash")},
                    "helm_input_values_changed": before.get("values") != after.get("values")}
        resources.append(resource)
    target = plan.get("planned_values", {}).get("outputs", {}).get("deployment_target", {}).get("value", {})
    enabled = target.get("api_enabled")
    output = {"summary": counts, "resources": resources,
              "planned_api_enabled": enabled if isinstance(enabled, bool) else "[unknown/unexpected]",
              "raw_values_and_provider_metadata": "omitted"}
    output_changes = plan.get("output_changes", {}).values()
    output["no_changes"] = not any(counts.values()) and all(
        item["change"]["actions"] == ["no-op"] for item in plan.get("resource_changes", [])
    ) and all(item["actions"] == ["no-op"] for item in output_changes)
    return output


def main():
    result = review(json.load(sys.stdin))
    print(json.dumps(result, indent=2))
    if result["no_changes"]:
        print("No changes. Your infrastructure matches the configuration.")
    else:
        counts = result["summary"]
        print(f"Plan: {counts['add']} to add, {counts['change']} to change, {counts['destroy']} to destroy.")


if __name__ == "__main__":
    main()
