import ROOT
import matplotlib.pyplot as plt
import numpy as np
import math
import os

HNL_mass = "50GeV"
output_dir = HNL_mass + "_ejj_50k/sum_plots/"

input_path_Dirac = "/afs/cern.ch/user/d/dimoulin/FCCAnalyses_new/FCC-LLP/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/Analysis/outputs/HNL_Dirac_ejj_50GeV_1e-3Ve/output_stage1/HNL_Dirac_ejj_50GeV_1e-3Ve.root"
input_path_Majorana = "/afs/cern.ch/user/d/dimoulin/FCCAnalyses_new/FCC-LLP/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/Analysis/outputs/HNL_Majorana_ejj_50GeV_1e-3Ve/output_stage1/HNL_Majorana_ejj_50GeV_1e-3Ve.root"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

input_file_Dirac = ROOT.TFile.Open(input_path_Dirac, "READ")
input_file_Majorana = ROOT.TFile.Open(input_path_Majorana, "READ")

print("Files and events were correctly loaded")

# list of lists
# each internal list: hist name, x title, y title, xmin, xmax, nbins
variables_list = [
     ["RecoJet_e", "RecoJet_energy [GeV]", "Entries", 0, 70, 50],
     ["RecoJet_pt", "RecoJet_pt [GeV]", "Entries", 0, 70, 50],
     ["selectedJet_e", "selectedJet_energy [GeV]", "Entries", 0, 70, 50],
     ["selectedJet_pt", "selectedJet_pt [GeV]", "Entries", 20, 70, 50]
]

# File information : [File, 'legend', 'label']
file_info = [
    [input_file_Majorana , 'Majorana ' + HNL_mass + ' semi-leptonic', 'Majorana'],
    [input_file_Dirac , 'Dirac ' + HNL_mass + ' semi-leptonic', 'Dirac']
]

sum_var = 0
def make_list(_file, variables_list):
	final_list = []
	events = _file.Get("events")
	nEntries = events.GetEntries()
	for variable in variables_list:
		var_list = []
		for i in range(0, nEntries):
			events.GetEntry(i)
			variable_list = list(getattr(events, variable[0]))
			for j in range(0, len(variable_list)):
				if len(variable_list) == 0:
					sum_var = 0
				else:
					sum_var = sum(variable_list)
			var_list.append(sum_var)
		final_list.append(var_list)
	return final_list			
print("Building lists ...")
Dirac_list = make_list(input_file_Dirac, variables_list)
Majorana_list = make_list(input_file_Majorana, variables_list)
print('Dirac : number of lists : ', len(Dirac_list), ' number of entries per list : ', len(Dirac_list[0]))
print('Majorana : number of lists : ', len(Majorana_list), ' number of entries per list : ', len(Majorana_list[0]))

def make_plot(Dirac_list, Majorana_list, variables_list):
	for i, variable in enumerate(variables_list):
		print('Looking at variable ' + variable[0])
		xmin = variable[3]
		xmax = variable[4]
		nbins = variable[5]

		#weights_Dirac =  np.ones_like(Dirac_list[i]) / len(Dirac_list[i])
		#weights_Majorana =  np.ones_like(Majorana_list[i]) / len(Majorana_list[i])
		fig, ax = plt.subplots()
		plt.hist(Dirac_list[i], bins = nbins, density=True, histtype = 'step', label = 'Dirac')
		plt.hist(Majorana_list[i], bins = nbins, density=True, histtype = 'step', label = 'Majorana')
		ax.set_xticks(np.arange(xmin, xmax, step = 5), minor=True)
		plt.xlabel("Total " + variable[1] + ' per event')
		plt.ylabel("Entries")
		plt.legend(loc = 'upper right')
		plt.savefig(output_dir + variable[0] + '_sum.png')
		print('Histogram for variable ' + variable[0] + ' saved')
		plt.close()

make_plot(Dirac_list, Majorana_list, variables_list)







'''
Dirac_events = input_file_Dirac.Get("events")
Dirac_nEntries = Dirac_events.GetEntries()
for i in range(0, 10):
	Dirac_events.GetEntry(i)
	energy =  list(getattr(Dirac_events, "RecoJet_e"))
	for j in range(len(energy)+1):
		if len(energy) == 0:
			sum_e = 0
		else:
			sum_e = sum(energy)
	print(i, energy," | Energy sum : ", sum_e)
print('-------------------------------------------')

Dirac_RecoJet_sumE = []
for i in range(0, Dirac_nEntries):
	Dirac_events.GetEntry(i)
	energy = list(getattr(Dirac_events, "RecoJet_e"))
	for j in range(len(energy) +1):
		if len(energy) == 0:
			sum_energy = 0
		else:
			sum_energy = sum(energy)
	Dirac_RecoJet_sumE.append(sum_energy)
print(Dirac_RecoJet_sumE[:20])
print('-------------------------------------------')
Dirac_RecoJet_sumE_clear = [i for i in Dirac_RecoJet_sumE if i != 0]
print('no zero :', Dirac_RecoJet_sumE_clear[:20])
print('-------------------------------------------')
print("Number of entries: ", Dirac_nEntries)
print("Size of sum_E: ", len(Dirac_RecoJet_sumE))
print("Size of sum_E without 0 jets: ", len(Dirac_RecoJet_sumE_clear))

bins = np.linspace(math.ceil(min(Dirac_RecoJet_sumE_clear)), math.floor(max(Dirac_RecoJet_sumE_clear)), 50)
plt.hist(Dirac_RecoJet_sumE_clear, bins=bins, histtype = 'step')
plt.title('Total jet energy per event')
plt.xlabel('RecoJet energy [GeV]')
plt.ylabel('Entries')
plt.legend('Dirac', loc='upper right')
plt.savefig(output_dir + 'RecoJet_sumE.png')
print('Histogram saved')

	
	#RecoElectron_energy.append(list(getattr(events, "RecoElectron_e")))
#print(RecoElectron_energy[:50])
#rows = len(RecoElectron_energy)
#column = len(RecoElectron_energy[0])
#print("Rows: ", rows, " Column :", column)	
#print("Shape : ", len(RecoElectron_energy)) 

'''
