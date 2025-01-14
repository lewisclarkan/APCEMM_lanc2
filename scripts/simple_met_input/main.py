from src.met import make_idealised_met
from src.ISA import ISA
import matplotlib.pyplot as plt

######################### MAIN FUNCTION (EXAMPLE) #########################

if __name__ == "__main__":
    met = make_idealised_met(
        centre_altitude_m = ISA().compute_h_from_p(23850), # m
        moist_depth_m = 500, # m
        RHi_background_PC = 20, # m
        RHi_peak_PC = 110, # %
        grad_RHi_PC_per_m = None, # % RHi / m; None indicates a rectangular moist region
        alt_resolution_m = 50, # m
        upper_alt_m = 11001, # m
        shear_over_s = 4e-3, # 1/s
        T_offset_K = 100, # K
        filename_prefix = "example"
    )

    RHi_PC = met.relative_humidity_ice.to_numpy()
    alt_m = met.altitude.to_numpy() * 1e3 # km to m

    plt.plot(RHi_PC, alt_m, color = 'b', lw = 1.5)
    plt.ylabel("Altitude, m")
    plt.xlabel('RHi, %')
    #plt.yticks([10250, 10500, 10750, 11000])
    plt.ylim(9000,11000)
    plt.xlim(0,150)
    plt.savefig('outputs/e.jpg')


    
