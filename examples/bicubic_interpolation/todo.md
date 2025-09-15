


**Root finding fro other properties than p-h**
PROPERTY_CALCULATORS = {
    fp.PT_INPUTS: jit_PT,
    fp.HmassSmass_INPUTS: jit_hs,
    fp.HmassP_INPUTS: jit_hP,
    fp.PSmass_INPUTS: jit_Ps,
    fp.DmassHmass_INPUTS: jit_rhoh,
    fp.DmassP_INPUTS: jit_rhop,
}

def get_props(input_pair, prop1, prop2, constants):
    return PROPERTY_CALCULATORS[input_pair](prop1, prop2, constants)


we specify the input_pair as cpx.HmassP_INPUTS and the table object as inputs


(these others need a 1d root finding)

Would it be possible to clean and use the functions:

def inverse_interpolant_scalar_DP(D, P):
def inverse_interpolant_scalar_hD(h, D):

or should we implement a generic root finder using optimistix? (fzero-like function that is compatible with jax)


**Jax verification**
Another element that we miss is one demo showing that the partial derivatives calculated with JAX agree with what is obtained from finite difference, or directly comparing with the bicubic polynomial coefficients directly


**Systematic testing with pytest**
Add automated testing to show that the error between Coolprop and bibuci is below a tolerance for a number of fluids and a number of states
Implement for a single fluid and state, and then we can add the parametrization for the other fluids and states.
