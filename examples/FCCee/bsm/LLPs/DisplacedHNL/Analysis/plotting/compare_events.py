import ROOT
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import itertools

HNL_mass = "50GeV"
output_dir = HNL_mass + "_correlation_plots/"

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

variables_list = [
     #["RecoDiJetElectron_invMass", "RecoDiJetElectron_invMass", "Entries", 0, 80, 20],
     #["GenDiJetElectron_invMass", "GenDiJetElectron_invMass", "Entries", 0, 80, 20],
     #["n_GenJets", "n_GenJets", "Entries", -0.5, 10.5, 11],
     #["n_RecoJets", "n_RecoJets", "Entries", -0.5, 10.5, 11]
     ["Reco_selectedJet_n", "Reco_selectedJet_n", "Entries", -0.5, 3.5, 4],
     ["Gen_selectedJet_n", "Gen_selectedJet_n", "Entries", -0.5, 3.5, 4]
]

# File information : [File, 'legend', 'label']
file_info = [
    [input_file_Dirac, 'Dirac_ ' + HNL_mass + ' semi-leptonic', 'Dirac'],
    [input_file_Majorana , 'Majorana ' + HNL_mass + ' semi-leptonic', 'Majorana']
]


def make_list(_file, variables_list):
	final_list = []
	events = _file.Get("events")
	nEntries = events.GetEntries()
	print("Looking at file containing: ",nEntries, " events." )
	for variable in variables_list:
		var_list = []
		print("Looking at variable: ", variable[0])
		for i in range(0, nEntries):
			events.GetEntry(i)
			event = getattr(events, variable[0])
			var_list.append(event)
		final_list.append(var_list)
		print("Finished with ", variable[0])
	return final_list


print("Building lists ...")
Dirac_list = make_list(input_file_Dirac, variables_list)
#Majorana_list = make_list(input_file_Majorana, variables_list)

print("Element '0, 0' Dirac: ", Dirac_list[0][0])
print("Element '1, 0' Dirac: ", Dirac_list[1][0])

def make_plot(Dirac_list, variables_list):
	print('Looking at variable ' + variables_list[0][1])
	xmin = variables_list[0][3]
	xmax = variables_list[0][4]
	nbins = variables_list[0][5]
	#weights_Dirac =  np.ones_like(Dirac_list[i]) / len(Dirac_list[i])
	#weights_Majorana =  np.ones_like(Majorana_list[i]) / len(Majorana_list[i])
	fig, ax = plt.subplots()
	plt.hist2d(Dirac_list[0], Dirac_list[1], bins=(nbins), range=[[variables_list[0][3], variables_list[0][4]], [variables_list[1][3], variables_list[1][4]]], density = True)
	ax.set_xticks(np.arange(0, 4, step = 1), minor=True)
	plt.xlabel(variables_list[0][0]) 
	plt.xlim(xmin, xmax)
	plt.ylim(xmin,xmax)
	plt.ylabel(variables_list[1][0])
	plt.savefig(output_dir + variables_list[0][1] + '_VS_' + variables_list[1][1] + '.png')
	print('Histogram for variable ' + variables_list[0][1] + '_VS_' + variables_list[1][1] + ' saved')
	plt.close()

make_plot(Dirac_list, variables_list)

#similar_list = [item for item in Dirac_list if item in Majorana_list]
#print("len similar list :", len(similar_list))
#print(similar_list[:5])
