"""Equivalent native requests must trace identically across Python hash seeds."""

import json
import os
from pathlib import Path
import subprocess
import sys

import jaxprop as jp


def test_phase_support_trace_is_independent_of_python_hash_seed(tmp_path):
    source = jp.Fluid("Water", backend="HEOS")
    pc, hc = float(source.critical_point.p), float(source.critical_point.h)
    recipe = dict(
        name="Water",
        backend="HEOS",
        method="bicubic",
        table_options=dict(
            h_min=hc - 3e5,
            h_max=hc + 3e5,
            p_min=0.94 * pc,
            p_max=1.04 * pc,
            N_h=24,
            N_p=12,
            N_p_sat=24,
            coarse_step=1,
            phase_support=True,
            table_dir=str(tmp_path),
        ),
    )
    jp.Fluid(**recipe)
    script = """
import hashlib,json,sys
import jax,jax.numpy as jnp,jaxprop as jp
model=jp.Fluid(**json.loads(sys.argv[1]))
def values(x,model):
    state=model.get_state(jp.HmassP_INPUTS,x[0],x[1],errors='mask')
    return jnp.stack((state.T,state.rho,state.s,state.quality_mass))
ir=jax.jit(values).lower(jnp.array(json.loads(sys.argv[2])),model).compiler_ir().operation.get_asm(enable_debug_info=False)
print('TRACE_SHA256='+hashlib.sha256(ir.encode()).hexdigest())
"""
    hashes = []
    for seed in ("1", "2"):
        env = os.environ.copy()
        env.update(
            PYTHONHASHSEED=seed,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            JAX_ENABLE_X64="1",
            JAX_PLATFORMS="cpu",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                json.dumps(recipe),
                json.dumps([hc - 2e4, 0.99 * pc]),
            ],
            cwd=Path(jp.__file__).resolve().parent.parent,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        hashes.append(
            next(
                line.split("=", 1)[1]
                for line in result.stdout.splitlines()
                if line.startswith("TRACE_SHA256=")
            )
        )
    assert hashes[0] == hashes[1]
