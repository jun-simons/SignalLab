from __future__ import annotations


def trace_to_markdown(trace) -> str:
    lines = ["# Signal Lab Trace Report", ""]
    for stage in trace.stages:
        lines.append(f"## Stage {stage.index}: {stage.operation}")
        lines.append("")
        lines.append(f"- Name: `{stage.name}`")
        lines.append(f"- Duration: `{stage.duration:.6g}` s")
        lines.append(f"- Samples: `{stage.n_samples}`")
        lines.append(f"- Sample rate: `{stage.sample_rate:.6g}` Hz")
        lines.append(f"- RMS: `{stage.rms:.6g}`")
        lines.append(f"- Peak: `{stage.peak:.6g}`")
        if stage.params:
            lines.append(f"- Parameters: `{stage.params}`")
        if stage.warnings:
            lines.append("- Warnings:")
            for warning in stage.warnings:
                lines.append(f"  - **{warning.code}**: {warning.message}")
        lines.append("")
    return "\n".join(lines)
