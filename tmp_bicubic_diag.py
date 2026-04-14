import numpy as np
import jaxprop as jxp

fluid_name='CO2'
backend='HEOS'
h_min=250e3
h_max=550e3
p_min=2.5e6
p_max=10e6
N_h=100
N_p=100

h_vals=np.linspace(h_min,h_max,N_h)
logP_vals=np.linspace(np.log(p_min),np.log(p_max),N_p)
fluid=jxp.Fluid(fluid_name,backend)

fails=[]
success=0
for i,h in enumerate(h_vals):
    for j,logP in enumerate(logP_vals):
        p=float(np.exp(logP))
        eps_h=1e-5*abs(h)
        eps_p=1e-5*abs(p)
        try:
            f0 = fluid.get_state(jxp.HmassP_INPUTS, h, p)
            fhp = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p)
            fhm = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p)
            fph = fluid.get_state(jxp.HmassP_INPUTS, h, p + eps_p)
            fpm = fluid.get_state(jxp.HmassP_INPUTS, h, p - eps_p)
            fh_p = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p + eps_p)
            fh_m = fluid.get_state(jxp.HmassP_INPUTS, h + eps_h, p - eps_p)
            fm_p = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p + eps_p)
            fm_m = fluid.get_state(jxp.HmassP_INPUTS, h - eps_h, p - eps_p)
            _ = f0['temperature'] + fhp['temperature'] + fhm['temperature'] + fph['temperature'] + fpm['temperature'] + fh_p['temperature'] + fh_m['temperature'] + fm_p['temperature'] + fm_m['temperature']
            success += 1
        except Exception as e:
            fails.append((i,j,h,p,str(e)))

print('success', success, 'fails', len(fails), 'total', N_h*N_p)
for rec in fails[:60]:
    i,j,h,p,msg=rec
    print(f'i={i:3d} j={j:3d} h={h:.6f} p={p:.6f}')
    print(msg.splitlines()[0])
print('--- last 10 ---')
for rec in fails[-10:]:
    i,j,h,p,msg=rec
    print(f'i={i:3d} j={j:3d} h={h:.6f} p={p:.6f}')
    print(msg.splitlines()[0])
