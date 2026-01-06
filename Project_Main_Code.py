import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema

print('This is a program to simulate the trajectory of a particle with negligible mass in the vicinity of a Schwarzschild black hole\n.')  
print('Here we use Geometrised units where everything is expressed in powers of length. Kindly make appropriate conversions as needed before using the code. \n')
M = float(input("Enter the mass of the black hole (M): "))
m = float(input("Enter the mass of the particle (m): "))
R_s = 2*M
def g(r):    
    gt = 1 - (R_s) / r
    gr = -1 / (1 - R_s / r)
    gphi = - r ** 2
    return gt, gr, gphi
def dUdtau(tau : float, X : np.ndarray):
    r, Ut, Ur, Uphi = X[1], *X[3:6] 
    gt = 1 - (R_s)/ r
    dUtdtau = - ((R_s) /((r)*(r-R_s))) * Ur * Ut
    dUrdtau = (R_s * (Ur)**2 )/(2*r*(R_s-r)) - (R_s*(r-R_s)*(Ut)**2 )/ (2*r**3) - (R_s - r)*(Uphi)**2
    dUphidtau = - 2 / r * Ur * Uphi
    return Ut, Ur, Uphi, dUtdtau, dUrdtau, dUphidtau
def horizon(tau : float, X : np.ndarray):
    return abs(round(X[1], 2) - R_s)
def trajectory(input_file : str, output_file : str):
    #pre-defined values 
    rtol = 1e-12
    atol = 1e-14
    method = 'Radau'
    component_names = ['t','r','phi','Ut','Ur','Uphi']   
    # stop the particle if it reaches the event horizon
    horizon.terminal = True    
    # reading in inputs as string and split into a list of seperate lines
    input_par = open(input_file, "r")
    inputs = input_par.read().replace(' ', '')
    input_par.close()
    inputlist = inputs.splitlines()
    # check each line in the input file for the initial conditions and return them as seperate variables as the required datatype    
    for con in inputlist:
        con_type = con[:con.find('=')]
        match con_type:
            case 'R0': 
                con = con.replace('R0=','')
                R0 = np.array(con.split(',')).astype(float)
            case 'U0':
                con = con.replace('U0=','')
                U0 = np.array(con.split(',')).astype(float)
            case 'Ntau':
                Ntau = int(float(con.replace('Ntau=','')))
            case 'tauf':
                tauf = float(con.replace('tauf=',''))
            case 'method':
                method = con.replace('method=','')
            case 'rtol':
                rtol = float(con.replace('rtol=',''))
            case 'atol':
                atol = float(con.replace('atol=',''))
            case 'rmax':
                rmax = float(con.replace('rmax=',''))
    # check that the neccesary input conditions are present in the file
    try: r0 = R0[1]
    except: raise Exception('Input file must contain a line with initial position \'R0 = t0, r0, phi0\'')
    try: Ur0, Uphi0 = U0
    except: raise Exception('Input file must contain a line with initial velocity \'U0 = Ur0, Uphi0\'')
    try: tauf
    except:  raise Exception('Input file must contain a line with proper time duration \'tauf = ')   
    try: tpoints = np.linspace(0, tauf, Ntau) # setup a grid of Ntau points over the proper time period [0, tauf]
    except:  raise Exception('Input file must contain a line with proper time duration \'Ntau = ')   
    # calculate the time component of the four velocity using the magnitude identity and the values of the schwarzschild metric at r0  
    gt0, gr0, gphi0 = g(r0)
    Ut0 = np.sqrt((1 - gr0 * Ur0 ** 2 - gphi0 * Uphi0 ** 2) / gt0)
    U0 = np.insert(U0, 0, Ut0)
    # perform the integration using the initial conditions
    X = solve_ivp(dUdtau, (0, tauf), (*R0, *U0), method = method, t_eval = tpoints, rtol = rtol, atol = atol, events = horizon)
    E = (1 - R_s/r0)* Ut0
    L = r0**2 * Uphi0
    
    print('The particle has an Initial Energy: ', E, 'units per unit mass')
    print('The particle has an Initial Angular Momentum: ', L, 'units per unit mass') 
    
    # Final values
    r_final = X.y[1][-1]
    Ut_final = X.y[3][-1]
    Uphi_final = X.y[5][-1]
    # Metric component at final radius
    gtf, _, gphif = g(r_final)
    # Final conserved quantities
    E_final = gtf * Ut_final
    L_final = (-gphif) * Uphi_final
    print('The particle has a Final Energy: ', E_final, 'units per unit mass')
    print('The particle has a Final Angular Momentum: ', L_final, 'units per unit mass')  
    Angle_Precess = 6 * np.pi * M**2 / L**2
    # Insert theoretical precession calculation here
    print(f"Precession Angle per Orbit: {round(Angle_Precess * 180 / np.pi, 12)} degrees")
    # --- OBSERVED PRECESSION CALCULATION ---
    r_values = X.y[1]
    phi_values = X.y[2]
    # Find local minima in r to detect periapsis points
    periapsis_indices = argrelextrema(r_values, np.less)[0]

    if len(periapsis_indices) >= 2:

        phi_at_periapses = phi_values[periapsis_indices]
        
        # Calculate angle differences (in radians then converted to degrees)
        delta_phis = np.diff(phi_at_periapses)
        observed_prec_angles = delta_phis * 180 / np.pi

        # Check if orbit is precessing
        
        if all(val > 0 for val in observed_prec_angles):
            print("The orbit is precessing (prograde).")
        elif all(val < 0 for val in observed_prec_angles):
            print("The orbit is precessing (retrograde).")
    else:
        print("Inconsistent or no clear precession detected.")
    # check if the integration ends due to the particle reaching the event horizon
    if X.status == 1: print('The test particle reaches the event horizon after proper time %.2f RS / c' % round(X.t[-1],2))
    # convert the returned arrays of proper time, four position and four velocity into strings to write to the output file
    out_string = 'tau' + ' = ' + ', '.join(X.t.astype(str)) + '\n'
    for y, f_type in zip(X.y, component_names):
        out_string += f_type + ' = ' + ', '.join(y.astype(str)) + '\n'
    # writes the proper time, four position and four velocity to the output file
    output_par = open(output_file,'w')
    output_par.write(out_string)
    # plots the azimuthal angular and radial coordinates on an 8*8 polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize = (8,8))
    plt.title("Trajectory of a massive particle around a Schwarzschild Black hole")
    ax.plot([], [], ' ', label=f'Mass of Black hole: M = {M}')
    # Plot the event horizon as a dashed circle
    event_horizon_radius = 2 * float(M)  # R_s = 2M in geometric units
    theta_vals = np.linspace(0, 2 * np.pi, 1000)
    r_vals = np.full_like(theta_vals, event_horizon_radius)
    ax.plot(theta_vals, r_vals, 'r--', label='Event Horizon (R_s = 2M)')
    # Optional: Add a legend and annotate the horizon
    ax.legend(loc='upper right')
    ax.annotate('Event Horizon', xy=(0, event_horizon_radius + 2), ha='center', color='red')
    ax.plot(X.y[2] , X.y[1])
    # checks if rmax was given as an input and if so sets the maximum radius of the polar plot
    try: ax.set_rmax(rmax)
    except: pass
    # labels the radial direction on the polar plot
    label_position = ax.get_rlabel_position()
    ax.text(np.radians(label_position - 10), ax.get_rmax() / 2,'Orbital radius [$R_\\text{S}$]', rotation = label_position, ha = 'center',va = 'center')
    # Plot the particle trajectory
    ax.plot(X.y[2], X.y[1], color='blue', label='Trajectory')
    ax.legend(loc='lower left')
    # Add a black dot at the initial position
    ax.plot(X.y[2][0], X.y[1][0], 'ko', label='Initial Position')  # 'ko' means black circle
    ax.legend(loc='upper right')
    plt.savefig("Trajectory_Plot.png")
    plt.show()
    # returns arrays of proper time, four position and four velocity
    return X.t, X.y[0:3], X.y[3:6]    
print('This is a program to simulate the trajectory of a particle with negligable mass in the vicinity of a black hole. Kindly Enter the necessary inputs.')
input_file = input("The file should contain the following parameters as: \n Initial position components(t,r,phi).\n Initial Velocity Components (Ur0,Uphi0). \n Ntau (Number of proper time points to evaluate R and U at). \n and Tauf (Length of proper time interval to model the trajectory). \n You may also include method,rtol,atol,rmax as the other specific paramters optionally as needed.\n Enter input file:\n ")
output_file = input("Enter output filename: \n")
_,_, _ = trajectory(input_file,output_file)
print("\n")
plt.show()

