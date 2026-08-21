"""EPM Python convenience entry points."""

# Resolved on first access rather than at import: `epm.geodata` builds the zone
# layers without any of the post-processing stack, and eagerly importing
# `run_data_inception_report` here would pull that stack (gams.transfer,
# matplotlib, seaborn) into every such build -- some six seconds for a name it
# never uses. `from epm import run_data_inception_report` still works.
def __getattr__(name):
    if name == "run_data_inception_report":
        from epm.postprocessing import run_data_inception_report
        return run_data_inception_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_data_inception_report"]
