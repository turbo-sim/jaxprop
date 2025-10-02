New files added:

Location: jaxprop/examples/bicubic_interpolation

Files:

1. demo_bisection.py
    A general bisection code. Not used in the latest code. CAN BE DELETED

2. demo_interpolation_midpoints_update.py   
    An update of demo_interpolation_midpoints to test the updated FluidBicubic output with FluidJax output at 
    the midpoints of the grid. POSITIVELY WORKING with the latest code.

3. demo_solver_consistency_copy.py
    A copy of demo_solver_consistency to extend the tests for all calls. Not used in the latest code.
    CAN BE DELETED

4. demo_solver_consistency_update.py 
    An update of demo_solver_consistency to test the updated FluidBicubic solver consistency with all calls.
    POSITIVELY WORKING with the latest code.

5. demo_table_interpolation_update.py
    An update of demo_table_interpolation to test the updated FluidBicubic output with FluidJAX output at random 
    pressure enthalpy point in the grid space. POSITIVELY WORKING with the latest code.

Location: jaxprop/jaxprop/bicubic

Files:

1. bicubic_interpolation_update.py
    An update of bicubic_interpolation. Doesn't work with Aliases of Property

2. bicubic_interpolation_update2.py
    An update of bicubic_interpolation. Works with all Property aliases and defines the new FluidBicubic class.
    POSITIVELY WORKING with the latest code.