"""State collection traces must not depend on Python's randomized hash seed."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("operation",["stack","concatenate"])
def test_state_collection_stablehlo_is_identical_across_hash_seeds(operation):
    root=Path(__file__).resolve().parents[1]
    program=r'''
import hashlib,json
import jax
import jax.numpy as jnp
import jaxprop as jp
jax.config.update("jax_enable_x64",True)
model=jp.Fluid("Air",method="perfect_gas")
def evaluate(x,fluid):
    a=fluid.get_state(jp.HmassP_INPUTS,x[0],x[1],errors="mask")
    b=fluid.get_state(jp.HmassP_INPUTS,x[0]*1.1,x[1]*1.2,errors="mask")
    state=jp.FluidState.stack((a,b)) if OPERATION=="stack" else a+b
    values=jnp.stack((state.p,state.h,state.T,state.s,state.rho))
    return values,jnp.all(jnp.isfinite(values)) & jnp.all(state.status==0)
function=jax.jit(evaluate)
x=jnp.array([500000.,100000.])
ir=function.lower(x,model).compiler_ir().operation.get_asm(enable_debug_info=False)
values,valid=function(x,model)
assert bool(valid)
print(json.dumps(dict(sha256=hashlib.sha256(ir.encode()).hexdigest(),values=values.tolist())))
'''.replace("OPERATION",repr(operation))
    results=[]
    for seed in ("1","2"):
        environment=os.environ.copy()
        environment.update(PYTHONHASHSEED=seed,PYTHONPATH=str(root),JAX_DISABLE_JIT="false")
        completed=subprocess.run([sys.executable,"-c",program],env=environment,
            capture_output=True,text=True,check=True,timeout=120)
        results.append(json.loads(completed.stdout.splitlines()[-1]))
    assert results[0]==results[1]
