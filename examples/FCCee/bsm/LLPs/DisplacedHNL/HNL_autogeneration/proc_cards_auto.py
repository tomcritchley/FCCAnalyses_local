import os

masses = [10, 20, 30, 40, 50, 60, 70, 80]
angles = [('1e-2', 0.01), ('1e-2.5', 0.0031622776601683795), ('1e-3', 0.001),
          ('1e-3.5', 0.00031622776601683794), ('1e-4', 0.0001), ('1e-4.5', 3.1622776601683795e-5),
          ('1e-5', 1e-5)]

def format_angle(angle):
    return str(angle).replace('.', 'p')

for mass in masses:
    for angle_sci, angle_dec in angles:
        filename = f"mg5_proc_card_HNL_Dirac_ejj_{mass}GeV_{format_angle(angle_sci)}Ve.dat"
        content = f"""\
#************************************************************
#*                     MadGraph5_aMC@NLO                    *
#*                                                          *
#*                *                       *                 *
#*                  *        * *        *                   *
#*                    * * * * 5 * * * *                     *
#*                  *        * *        *                   *
#*                *                       *                 *
#*                                                          *
#*                                                          *
#*         VERSION 2.7.3                 2020-06-21         *
#*                                                          *
#*    The MadGraph5_aMC@NLO Development Team - Find us at   *
#*    https://server06.fynu.ucl.ac.be/projects/madgraph     *
#*                                                          *
#************************************************************
#*                                                          *
#*               Command File for MadGraph5_aMC@NLO         *
#*                                                          *
#*     run as ./bin/mg5_aMC  filename                       *
#*                                                          *
#************************************************************
#This card was originally made by suchita Kulkarni
#contact suchita.kulkarni@cern.ch
#Then edited by Dimitri Moulin
#contact dimitri.moulin@etu.unige.ch
set default_unset_couplings 99
set group_subprocesses Auto
set ignore_six_quark_processes False
set loop_optimized_output True
set loop_color_flows False
set gauge unitary
set complex_mass_scheme False
set max_npoint_for_channel 0
import model sm
define p = g u c d s u~ c~ d~ s~
define j = g u c d s u~ c~ d~ s~
define l+ = e+ mu+
define l- = e- mu-
define vl = ve vm vt
define vl~ = ve~ vm~ vt~
import model SM_HeavyN_Dirac_CKM_Masses_LO
define e = e+ e-
define nue = ve ve~
generate e+ e- > n1~ ve , (n1~ > e+ j j)
add process e+ e- > n1 ve~ , (n1 > e- j j )
output HNL_Dirac_ejj_{mass}GeV_{format_angle(angle_sci)}Ve
launch HNL_Dirac_ejj_{mass}GeV_{format_angle(angle_sci)}Ve
done
# set to electron beams (0 for ele, 1 for proton)
set lpp1 0
set lpp2 0
set ebeam1 45.594
set ebeam2 45.594
set no_parton_cut
# Here I set mass of the electron HNL
set mn1 {mass}
# set mass of muon HNL, made heavy here
set mn2 10000
# set mass of tau HNL, made heavy here
set mn3 10000
# set electron mixing angle
set ven1 {angle_dec}
# this is important, set the decay width of HNL flavour of interest to auto
# if this is not done, lifetime calculations won'e be right
set WN1 auto
set time_of_flight 0
set nevents 100000
done
"""
        # Write to file
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Generated: {filename}")

print("Process cards generation complete.")
