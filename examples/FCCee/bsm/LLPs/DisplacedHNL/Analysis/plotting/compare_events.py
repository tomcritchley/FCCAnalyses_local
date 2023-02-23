import ROOT
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import itertools

HNL_mass = "50GeV"
output_dir = HNL_mass + "_correlation_plots/"

input_path_Dirac = "/afs/cern.ch/user/d/dimoulin/FCCAnalyses_new/FCC-LLP/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/Analysis/outputs/HNL_Dirac_ejj_50GeV_1e-3Ve/output_stage1/HNL_Dirac_ejj_50GeV_1e-3Ve.root"
#input_path_Majorana = "/afs/cern.ch/user/d/dimoulin/FCCAnalyses_new/FCC-LLP/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/Analysis/outputs/HNL_Majorana_ejj_50GeV_1e-3Ve/output_stage1/HNL_Majorana_ejj_50GeV_1e-3Ve.root"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

input_file_Dirac = ROOT.TFile.Open(input_path_Dirac, "READ")
#input_file_Majorana = ROOT.TFile.Open(input_path_Majorana, "READ")


reco_variables_list = [
      ["RecoLeadJet_e", "LeadJet_e", "Entries", 0, 50, 20],
      ["RecoLeadJet_pt", "LeadJet_pt", "Entries", 0, 50, 20],
      ["RecoLeadJet_eta", "LeadJet_eta", "Entries", -3.2, 3.2, 20],
      ["RecoLeadJet_phi", "LeadJet_phi", "Entries", -3.2, 3.2, 20],

      ["RecoSecondJet_e", "SecondJet_e", "Entries", 0, 50, 20],
      ["RecoSecondJet_pt", "SecondJet_pt", "Entries", 0, 50, 20],
      ["RecoSecondJet_eta", "SecondJet_eta", "Entries", -3.2, 3.2, 20],
      ["RecoSecondJet_phi", "SecondJet_phi", "Entries", -3.2, 3.2, 20],


      ["RecoDiJetElectron_invMass", "RecoDiJetElectron_invMass", "Entries", 0, 80, 20],

      ["n_RecoJets", "n_Jets", "Entries", -0.5, 10.5, 11],

      ["RecoDiJet_e", "DiJet_e", "Entries", 0, 70, 20],
      ["RecoDiJet_pt", "DiJet_pt", "Entries", 0, 50, 20],
      ["RecoDiJet_eta", "DiJet_eta", "Entries", -3.2, 3.2, 20],
      ["RecoDiJet_phi", "DiJet_phi", "Entries", -3.2, 3.2, 20],

      ["RecoElectron_lead_e", "Electron_lead_e", "Entries", 0, 70, 20],
      ["RecoElectron_lead_pt", "Electron_lead_pt", "Entries", 0, 50, 20],
      ["RecoElectron_lead_eta", "Electron_lead_eta", "Entries", -3.2, 3.2, 20],
      ["RecoElectron_lead_phi", "Electron_lead_phi", "Entries", -3.2, 3.2, 20],

      ["RecoDiJetElectron_e", "DiJetElectron_e", "Entries", 0, 100, 20],
      ["RecoDiJetElectron_pt", "DiJetElectron_pt", "Entries", 0, 80, 20],
      ["RecoDiJetElectron_eta", "DiJetElectron_eta", "Entries", -3.2, 3.2, 20],
      ["RecoDiJetElectron_phi", "DiJetElectron_phi", "Entries", -3.2, 3.2, 20],

]

gen_variables_list = [
      ["GenLeadJet_e", "LeadJet_e", "Entries", 0, 50, 20],
      ["GenLeadJet_pt", "LeadJet_pt", "Entries", 0, 50, 20],
      ["GenLeadJet_eta", "LeadJet_eta", "Entries", -3.2, 3.2, 20],
      ["GenLeadJet_phi", "LeadJet_phi", "Entries", -3.2, 3.2, 20],

      ["GenSecondJet_e", "SecondJet_e", "Entries", 0, 50, 20],
      ["GenSecondJet_pt", "SecondJet_pt", "Entries", 0, 50, 20],
      ["GenSecondJet_eta", "SecondJet_eta", "Entries", -3.2, 3.2, 20],
      ["GenSecondJet_phi", "SecondJet_phi", "Entries", -3.2, 3.2, 20],


      ["GenDiJetElectron_invMass", "GenDiJetElectron_invMass", "Entries", 0, 80, 20],

      ["n_GenJets", "n_GenJets", "Entries", -0.5, 10.5, 11],

      ["GenDiJet_e", "GenDiJet_e", "Entries", 0, 70, 20],
      ["GenDiJet_pt", "GenDiJet_pt", "Entries", 0, 50, 20],
      ["GenDiJet_eta", "GenDiJet_eta", "Entries", -3.2, 3.2, 20],
      ["GenDiJet_phi", "GenDiJet_phi", "Entries", -3.2, 3.2, 20],

      ["GenHNLElectron_e", "GenHNLElectron_e", "Entries", 0, 70, 20],
      ["GenHNLElectron_pt", "GenHNLElectron_pt", "Entries", 0, 50, 20],
      ["GenHNLElectron_eta", "GenHNLElectron_eta", "Entries", -3.2, 3.2, 20],
      ["GenHNLElectron_phi", "GenHNLElectron_phi", "Entries", -3.2, 3.2, 20],

      ["GenDiJetElectron_e", "GenDiJetElectron_e", "Entries", 0, 100, 20],
      ["GenDiJetElectron_pt", "GenDiJetElectron_pt", "Entries", 0, 80, 20],
      ["GenDiJetElectron_eta", "GenDiJetElectron_eta", "Entries", -3.2, 3.2, 20],
      ["GenDiJetElectron_phi", "GenDiJetElectron_phi", "Entries", -3.2, 3.2, 20],

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
Dirac_reco_list = make_list(input_file_Dirac, reco_variables_list)
Dirac_gen_list = make_list(input_file_Dirac, gen_variables_list)

#Dirac_list = make_list(input_file_Dirac, variables_list)
#Majorana_list = make_list(input_file_Majorana, variables_list)

#print("Element '0, 0' Dirac: ", Dirac_list[0][0])
#print("Element '1, 0' Dirac: ", Dirac_list[1][0])

def make_plot_2d(Dirac_reco_list, Dirac_gen_list, reco_variables_list, gen_variables_list):
	for i, variable in enumerate(reco_variables_list):
		print('Preparing to plot variables: ', reco_variables_list[i][0], " and ", gen_variables_list[i][0])
		xmin = reco_variables_list[i][3]
		xmax = reco_variables_list[i][4]
		nbins = reco_variables_list[i][5]
		print('xmin = ', xmin, ' xmax = ', xmax, ' nbins = ', nbins)
		fig, ax = plt.subplots()
		print('Plotting histograms..')
		plt.hist2d(Dirac_reco_list[i], Dirac_gen_list[i], bins= nbins, range = [[xmin,xmax], [xmin,xmax]], density= True)
		ax.set_xticks(np.arange(xmin, xmax, step = 5), minor = True)
		plt.xlabel(reco_variables_list[i][0])
		plt.ylabel(gen_variables_list[i][0])
		plt.xlim(xmin,xmax)
		plt.ylim(xmin,xmax)
		plt.savefig(output_dir + '2D_'+ reco_variables_list[i][0] + '_VS_' + gen_variables_list[i][0] + '.png')
		print('Histogram for variable ' + '2D_'+ reco_variables_list[i][0] + '_VS_' + gen_variables_list[i][0] + ' saved.')
		plt.close()


def make_plot_1d(Dirac_reco_list, Dirac_gen_list, reco_variables_list, gen_variables_list):
	for i, variable in enumerate(reco_variables_list):
		print('Preparing to plot variables: ', reco_variables_list[i][0], " and ", gen_variables_list[i][0])
		xmin = reco_variables_list[i][3]
		xmax = reco_variables_list[i][4]
		nbins = reco_variables_list[i][5]
		print('xmin = ', xmin, ' xmax = ', xmax, ' nbins = ', nbins)
		fig, ax = plt.subplots()
		print('Plotting histograms..')
		plt.hist(Dirac_reco_list[i], bins= nbins, label = 'Reco', histtype = 'step', density= True, range = (xmin, xmax), color = 'dodgerblue')
		plt.hist(Dirac_gen_list[i], bins= nbins, label = 'Gen', histtype = 'step', density= True, range = (xmin,xmax), color = 'magenta')
		ax.set_xticks(np.arange(xmin, xmax, step = 5), minor = True)
		plt.xlabel(reco_variables_list[i][1])
		plt.ylabel(reco_variables_list[i][2])
		plt.legend(loc = 'upper right')
		plt.xlim(xmin,xmax)
		plt.savefig(output_dir + '1D_'+ reco_variables_list[i][0] + '_VS_' + gen_variables_list[i][0] + '.png')
		print('Histogram for variable ' + '1D_' + reco_variables_list[i][0] + '_VS_' + gen_variables_list[i][0] + ' saved.')
		plt.close()

make_plot_2d(Dirac_reco_list, Dirac_gen_list, reco_variables_list, gen_variables_list)

make_plot_1d(Dirac_reco_list, Dirac_gen_list, reco_variables_list, gen_variables_list)

#similar_list = [item for item in Dirac_list if item in Majorana_list]
#print("len similar list :", len(similar_list))
#print(similar_list[:5])
