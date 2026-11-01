"""Identify the verbatim original files without editing their code."""
from pathlib import Path
import inspect
from .io import read, sha


def load_original():
    """Return actual imported modules and pinned source evidence."""
    from .original import tmcmc_mod, pdfs
    package = Path(__file__).resolve().parent
    reference = read(package / "SAMPLER_REFERENCE.json")
    evidence = {}
    for filename, module in [("tmcmc_mod.py", tmcmc_mod), ("pdfs.py", pdfs)]:
        expected = package.parent / reference[filename]["package_path"]
        imported = Path(module.__file__).resolve()
        if imported != expected.resolve() or sha(imported) != reference[filename]["sha256"]:
            raise RuntimeError(f"Original source/path verification failed: {filename}")
        evidence[filename] = {**reference[filename], "actual_import": str(imported), "module": module.__name__}
    if tmcmc_mod.run_tmcmc.__defaults__ != (5, 10):
        raise RuntimeError("Original sampler mutation defaults differ from the pinned source")
    if Path(tmcmc_mod.run_tmcmc.__code__.co_filename).resolve() != Path(tmcmc_mod.__file__).resolve():
        raise RuntimeError("Sampler function is not from the verified original module")
    if Path(pdfs.Uniform.log_pdf_eval.__code__.co_filename).resolve() != Path(pdfs.__file__).resolve():
        raise RuntimeError("Prior function is not from the verified original module")
    evidence["sampler_signature"] = str(inspect.signature(tmcmc_mod.run_tmcmc))
    return tmcmc_mod, pdfs, evidence
