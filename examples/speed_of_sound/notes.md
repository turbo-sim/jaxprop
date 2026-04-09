

mixture speed of sound in the full non-equilibrium model

Frozen-equation two-phase mixture




## Frozen--equation mixture speed of sound

In the non-equilibrium two-phase mixture, the phases have
independent pressures, temperatures and chemical potentials, but share a
common velocity. The mixture speed of sound $c_0$ follows directly from linearization
of the basic relaxation model and is given by
$$
    \tilde{a}_0^2
    = 
    y_{g}, a_g^2 + y_{\ell}\, a_\ell^2 ,
$$
where $y_g = \rho_g \alpha_g / \rho$ is the mass fraction of the gas phase and
$c_k$ is the isentropic sound speed of phase $k\in\{g,\ell\}$:
$$
    a_k^2 = \left( \frac{\partial p_k}{\partial \rho_k} \right)_{s_k}
$$

This expression represents the _frozen sound speed_, meaning that no
pressure, thermal or mass relaxation is allowed during acoustic propagation.
It is the highest‐valued mixture sound speed in the relaxation hierarchy.




## Pressure-equilibrium speed of sound

When the phases share the same pressure $p_g = p_\ell = p$, but retain
independent temperatures and chemical potentials, the acoustic propagation
follows the classical *Wood speed of sound*:

$$
\frac{1}{\rho \tilde{a}_p^2}
=
\frac{\alpha_g}{\rho_g a_g^2}
+
\frac{\alpha_\ell}{\rho_\ell a_\ell^2}
$$

where:
- $\alpha_k$ are the phase volume fractions,
- $\rho_k$ are the phase densities,
- $c_k$ are the isentropic sound speeds of the individual phases,
- $\rho = \alpha_g \rho_g + \alpha_\ell \rho_\ell$ is the mixture density.

This expression is always **lower** than the frozen speed of sound:
$$
\tilde{a}_p \le \tilde{a}_0.
$$



## Pressure–temperature equilibrium speed of sound

When both the pressure and temperature of the phases are relaxed  
$$
p_g = p_\ell = p, \qquad T_g = T_\ell = T,
$$
but the phases retain distinct chemical potentials, the mixture speed of sound
$\tilde{a}_{pT}$ is given by
$$
\frac{1}{\rho\,\tilde{a}_{pT}^2}
=
\frac{\alpha_g}{\rho_g\,a_g^2}
+
\frac{\alpha_\ell}{\rho_\ell\,a_\ell^2}
+
T\,
\frac{C_{p,g}\,C_{p,\ell}}{C_{p,g}+C_{p,\ell}}
\left(
    \frac{\Gamma_\ell}{\rho_\ell\,a_\ell^2}
    -
    \frac{\Gamma_g}{\rho_g\,a_g^2}
\right)^2.
$$

where:

- $C_{p,k} = \rho_k\,\alpha_k\,c_{p,k}$ are the extensive heat capacities,  
- $c_{p,k}$ is the specific isobaric heat capacity of phase $k$,  
- $\Gamma_k$ is the Grüneisen coefficient,  
- $a_k$ is the isentropic sound speed of phase $k$,  
- $\rho$ is the mixture density.

This relation can be expressed in incremental form as
$$
\tilde{a}_{pT}^{-2}
=
\tilde{a}_{p}^{-2}
+
Z^{p}_{pT},
$$
with the thermal–relaxation correction
$$
Z^{p}_{pT}
=
\rho\,T
\,
\frac{C_{p,g}\,C_{p,\ell}}{C_{p,g}+C_{p,\ell}}
\left(
    \frac{\Gamma_\ell}{\rho_\ell\,a_\ell^2}
    -
    \frac{\Gamma_g}{\rho_g\,a_g^2}
\right)^2.
$$

Since $Z^{p}_{pT} \ge 0$, the pressure–temperature relaxed sound speed satisfies
$$
\tilde{a}_{pT} \le \tilde{a}_{p} \le \tilde{a}_{0}.
$$



## Pressure–material equilibrium speed of sound

When the phases share a common pressure and chemical potential but still allow independent heat
transfer, the mixture follows the pressure–material equilibrium model. The corresponding sound speed $\tilde{a}_{p\mu}$ is given by:
$$
\frac{1}{\rho\,\tilde{a}_{p\mu}^2}
=
\frac{\alpha_g}{\rho_g\,a_g^2}
+
\frac{\alpha_\ell}{\rho_\ell\,a_\ell^2}
+ \frac{C_{p,g}\,C_{p,\ell}}
     {\rho_g^{\,2}\,\rho_\ell^{\,2}\,
      \left(C_{p,\ell}\,s_\ell^{\,2} T_\ell
           + C_{p,g}\,s_g^{\,2} T_g\right)}
\left(
\rho_g - \rho_\ell
+ \rho_g \rho_\ell
\left[
    s_g T_g\,\frac{\Gamma_g}{\rho_g a_g^{2}}
    -
    s_\ell T_\ell\,\frac{\Gamma_\ell}{\rho_\ell a_\ell^{2}}
\right]
\right)^{2}
$$
where:
- $C_{p,k} = \rho_k\,\alpha_k\,c_{p,k}$ are the extensive heat capacities,
- $c_{p,k}$ is the specific isobaric heat capacity of phase $k$,
- $s_k$ is the specific entropy,
- $T_k$ is the phase temperature,
- $\Gamma_k$ is the Grüneisen coefficient,
- $a_k$ is the isentropic sound speed of phase $k$,
- $\rho$ is the mixture density.


This relation can be expressed in incremental form as
$$
\tilde{a}_{p\mu}^{-2}
=
\tilde{a}_{p}^{-2}
+
Z^{p}_{p\mu}.
$$
The correction term $Z^{p}_{p\mu}$ is
$$
Z^{p}_{p\mu}
=
\frac{\rho\,C_{p,g}\,C_{p,\ell}}
     {\rho_g^{\,2}\,\rho_\ell^{\,2}\,
      \left(C_{p,\ell}\,s_\ell^{\,2} T_\ell
           + C_{p,g}\,s_g^{\,2} T_g\right)}
\left(
\rho_g - \rho_\ell
+ \rho_g \rho_\ell
\left[
    s_g T_g\,\frac{\Gamma_g}{\rho_g a_g^{2}}
    -
    s_\ell T_\ell\,\frac{\Gamma_\ell}{\rho_\ell a_\ell^{2}}
\right]
\right)^{2}.
$$

Since $Z^{p}_{p\mu} \ge 0$, the ordering of mixture sound speeds becomes
$$
\tilde{a}_{p\mu}

\;\le\;
\tilde{a}_{p}
\;\le\;
\tilde{a}_{0}.
$$


## Pressure–temperature–material equilibrium (pTμ) speed of sound

In the full relaxation limit, the phases are in equilibrium with respect to
pressure, temperature and chemical potential:
$$
p_g = p_\ell = p, \qquad
T_g = T_\ell = T, \qquad
\mu_g = \mu_\ell = \mu
$$
This corresponds to the homogeneous equilibrium model (HEM), where the mixture
behaves like a single effective fluid. The resulting speed of sound $\tilde{a}_{pT\mu}$ is given by
is given by:
$$
\frac{1}{\rho\,\tilde{a}_{p\mu T}^2}
=
\frac{\alpha_g}{\rho_g\,a_g^2}
+
\frac{\alpha_\ell}{\rho_\ell\,a_\ell^2}
+
T
\left[
\frac{\rho_g\,\alpha_g}{c_{p,g}}
\left( \frac{\partial s_g}{\partial p} \right)_{\!\mathrm{sat}}^{2}
+
\frac{\rho_\ell\,\alpha_\ell}{c_{p,\ell}}
\left( \frac{\partial s_\ell}{\partial p} \right)_{\!\mathrm{sat}}^{2}
\right].
$$
where:
- $\alpha_k$ are the phase volume fractions,
- $\rho_k$ are the phase densities,
- $c_{p,k}$ is the isobaric heat capacity,
- $s_k$ is the specific entropy,
- the derivative $(\partial s_k/\partial p)_{\mathrm{sat}}$ is evaluated along the *saturation curve*,
- $\rho$ is the mixture density.

This relation can be expressed in incremental form as
$$
\tilde{a}_{pT\mu}^{-2}
=
\tilde{a}_{p}^{-2}
+
Z^{p}_{pT\mu}.
$$

The correction term $Z^{p}_{pT\mu}$ is
$$
Z^{p}_{pT\mu}
=
\rho\,T
\left[
\frac{\rho_g\,\alpha_g}{c_{p,g}}
\left( \frac{\partial s_g}{\partial p} \right)_{\!\mathrm{sat}}^{2}
+
\frac{\rho_\ell\,\alpha_\ell}{c_{p,\ell}}
\left( \frac{\partial s_\ell}{\partial p} \right)_{\!\mathrm{sat}}^{2}
\right].
$$

Because $Z^{p}_{pT\mu} \ge 0$, the full ordering of relaxation sound speeds is
$$
\tilde{a}_{pT\mu}
\;\le\;
\tilde{a}_{pT}
\;\le\;
\tilde{a}_{p}
\;\le\;
\tilde{a}_{0}.
$$
as well as
$$
\tilde{a}_{pT\mu}
\;\le\;
\tilde{a}_{p\mu}
\;\le\;
\tilde{a}_{p}
\;\le\;
\tilde{a}_{0}.
$$

This is the lowest sound speed in the relaxation hierarchy, representing the
limit in which all interfacial exchange processes (pressure, heat, and phase
change) act infinitely fast, giving the mixture the acoustic response of a
single–phase effective medium.



> We review the wave structure of this model, and recover the well-known fact that this model has a discontinuous mixture sound velocity in the limit where one of the phases disappears. Remarkably, it turns out that this discontinuous behavior cannot be attributed to one single relaxation procedure; each procedure in itself yields continuous behavior. Rather, the discontinuity of the HEM model is an emergent phenomenon, arising only when all relaxation procedures are simultaneously applied.


These are the two phase-boundary jump conditions for the homogeneous equilibrium model:
$$
\lim_{\alpha_g \to 1} Z^{pT\mu}_{pT\mu}
\;=\;
\lim_{\alpha_g \to 1} Z^{p\mu}_{pT\mu}
\;=\;
c_{p,g}\,T
\left(
    \frac{\rho_g - \rho_\ell}{\rho_\ell\,(h_g - h_\ell)}
    + 
    \frac{\Gamma_g}{c_g^{2}}
\right)^{2}.
$$

$$
\lim_{\alpha_\ell \to 1} Z^{pT\mu}_{pT\mu}
\;=\;
\lim_{\alpha_\ell \to 1} Z^{p\mu}_{pT\mu}
\;=\;
c_{p,\ell}\,T
\left(
    \frac{\rho_g - \rho_\ell}{\rho_g\,(h_g - h_\ell)}
    +
    \frac{\Gamma_\ell}{c_\ell^{2}}
\right)^{2}.
$$

