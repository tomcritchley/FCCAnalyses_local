

import ROOT
from ROOT import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import itertools
import glob

intLumi =  1e+04 #in pb-1 (10 fb-1)

estimated_Err = 0.1 #estimated error on significance

log_scale = True

norm_hist = False #Normalize signal/background to 1

input_dir = "selected_hist/"
output_dir =  "Test_Generalized_Bg/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

selection_list = [
  #"selNone",
  #"selMissingEGt12",
  #"selMissingEGt12_EleEGt35",
  #"selMissingEGt12_EleEGt35_AngleLt24",
  #"selMissingEGt12_EleEGt25_EleSecondJetDRGt06",
  #"selMissingEGt12_EleEGt30_EleSecondJetDRGt08",
  #"selMissingEGt12_EleEGt35_EleSecondJetDRGt08",

  #"selMissingEGt12_EleEGt25_EleSecondJetDRGt06_DiJetDRLt3",
  #"selMissingEGt12_EleEGt30_EleSecondJetDRGt08_DiJetDRLt3",
  #"selMissingEGt12_EleEGt35_EleSecondJetDRGt08_DiJetDRLt3",
  "selMissingEGt12_EleEGt35_AngleLt24_DiJetEleDRLt3",

]

colors_signal = [876, 616, 880, 801, 629, 879, 602, 921, 622]
colors_bg = [801, 410, 856, 629, 879, 602, 921, 622]


def get_Z(n,b,e,b_thr=0):
  '''
  n: total number of events
  b: prediction of bkg
  e: error on the prediction
  '''
  if n<=0 or b<=b_thr:
    return 0
  ss=e*e  # sigma squared
  bb=b*b
  z=np.sqrt( 2*( n*np.log((n*b+n*ss)/(bb+n*ss)) - ((bb)/ss)*np.log(1+(ss*n-ss*b)/(bb+b*ss)) ))
  if n < b:
    return -z
  return z


def get_max_Z(h_signal, h_list_bg, x1, x2, stage1_path):
    #x1 : lower bound to compute significance
    #x2 : upper bound to compute the significance
    h_significance = h_signal.Clone("h_significance")
    h_significance_max = 0
    bin_x1 = h_significance.FindBin(x1)
    bin_x2 = h_significance.FindBin(x2)
    for bin in range(bin_x1, bin_x2):
        s = h_significance.Integral(bin, bin_x2)
        b = h_list_bg[0].Integral(bin, bin_x2) + h_list_bg[1].Integral(bin, bin_x2) + h_list_bg[2].Integral(bin, bin_x2)
        significance = 0
        n = s+b
        if n > 0:
            significance = get_Z(n, b, estimated_Err*b, 0)
        h_significance.SetBinContent(bin, significance)

        if h_significance.GetBinContent(bin) > h_significance_max:
            h_significance_max = h_significance.GetBinContent(bin)
    return h_significance_max

def get_max_significance(h_signal, h_list_bg, x1, x2, stage1_path):
    #x1 : lower bound to compute significance
    #x2 : upper bound to compute the significance
    h_significance = h_signal.Clone("h_significance")

    h_significance_max = 0
    bin_x1 = h_significance.FindBin(x1)
    bin_x2 = h_significance.FindBin(x2)
    for bin in range(bin_x1, bin_x2):
        s = h_significance.Integral(bin, bin_x2)
        b = h_list_bg[0].Integral(bin, bin_x2) + h_list_bg[1].Integral(bin, bin_x2) + h_list_bg[2].Integral(bin, bin_x2)
        significance = 0
        if s+b > 0:
            significance = s/sqrt(s+b)
        h_significance.SetBinContent(bin, significance)

        if h_significance.GetBinContent(bin) > h_significance_max:
            h_significance_max = h_significance.GetBinContent(bin)
    return h_significance_max


def make_hist(files_list):
	h_list = []
	n_events_list = []
	for f in files_list:
		print("Looking at file", f[1])
		my_file = ROOT.TFile.Open(f[0])
		print("Getting histogram for variable", f[1])
		hist = my_file.Get("RecoDiJetElectron_invMass")
		hist.SetDirectory(0)
		n_tot = get_n_events(f[3])
		n_events = hist.GetEntries()
		print("n_total:", n_tot)
		print("n_events:", n_events)
		xsec = f[2]
		hist.Scale(n_events/hist.Integral())
		print("Scale to n:", hist.Integral())
		x = (intLumi*xsec)/n_tot
		print("Scaling factor (L_int*N)/xsec =", x)
		hist.Scale(x)
		print("hist integral after scaling:", hist.Integral())
		n_events_list.append(n_events)
		h_list.append(hist)
		print("Histogram added to h_list")
		my_file.Close()
		print("-----------------------")
	return h_list, n_events_list 

def make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, selection):
    #Define Pads and legends
    c = ROOT.TCanvas("can","can",600,600)
    pad1 = ROOT.TPad("pad1", "pad1",0.0,0.0,1.0,1.0,21)
    pad1.SetFillColor(0)
    pad1.SetBottomMargin(0.1)
    if log_scale == True : pad1.SetLogy()
    pad1.SetTickx()
    pad1.SetTicky()
    pad1.Draw()

    leg_bg = ROOT.TLegend(0.7, 0.5, 0.9, 0.65)
    leg_bg.SetFillStyle(0)
    leg_bg.SetLineWidth(0)
    
    leg_sig = ROOT.TLegend(0.7, 0.70, 0.9, 0.85)
    leg_sig.SetFillStyle(0)
    leg_sig.SetLineWidth(0)
    
    #Compute significances 
    significance_20GeV = get_max_significance(h_list_signal[0], h_list_bg, 15, 25, "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_20GeV_1e-3Ve_W2023/output_stage1/")
    significance_50GeV = get_max_significance(h_list_signal[1], h_list_bg, 45, 55, "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_50GeV_1e-3Ve_W2023/output_stage1/")
    significance_70GeV = get_max_significance(h_list_signal[2], h_list_bg, 65, 75, "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_stage1/")

    #Compute Z
    Z_20GeV = get_max_Z(h_list_signal[0], h_list_bg, 15, 25, "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_20GeV_1e-3Ve_W2023/output_stage1/")
    Z_50GeV = get_max_Z(h_list_signal[1], h_list_bg, 45, 55, "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_50GeV_1e-3Ve_W2023/output_stage1/")
    Z_70GeV = get_max_Z(h_list_signal[2], h_list_bg, 65, 75, "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_stage1/")
 
    #Compute hist maximum
    h_list = h_list_signal + h_list_bg
    h_max = 0
    for ih,h in enumerate(h_list):
        if h.GetMaximum() > h_max:
            h_max = h.GetMaximum()
    for ih,h in enumerate(h_list_signal):
        leg_sig.AddEntry(h, legend_list_signal[ih])
    for ih,h in enumerate(h_list_bg):
        leg_bg.AddEntry(h, legend_list_bg[ih])

    # Draw in the top panel
    pad1.cd()
    #Draw signal histograms
    for ih,h in enumerate(h_list_signal):
        h.SetLineColor(colors_signal[ih])
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle("m [GeV]")
        h.GetYaxis().SetTitle("Normalised entries") if log_scale == False else h.GetYaxis().SetTitle("log Normalised entries")
        h.GetYaxis().SetTitleOffset(0.8)
        h.SetMaximum(1.25*h_max)
        if norm_hist == True :
            h.Scale(1.0 / h.Integral())
            h_max = 0
            if h.GetMaximum() > h_max:
                h_max = h.GetMaximum()
            h.SetMaximum(h_max*1.25)
        h.Draw('hist same')
     
    #Draw background histograms
    for ih,h in enumerate(h_list_bg):
        h.SetLineColor(colors_bg[ih])
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle("m [GeV]")
        h.GetYaxis().SetTitle("Normalised entries") if log_scale == False else h.GetYaxis().SetTitle("log Normalised entries")
        h.GetYaxis().SetTitleOffset(0.8)
        h.SetMaximum(1.25*h_max)
        if norm_hist == True :
            h.Scale(1.0 / h.Integral())
            h_max = 0
            if h.GetMaximum() > h_max:
                h_max = h.GetMaximum()
            h.SetMaximum(h_max * 1.25)
        h.Draw('hist same')

    leg_sig.Draw()
    leg_bg.Draw()
  
    #Plot informations 
    text_selection = ROOT.TLatex()
    text_selection.SetTextSize(0.04)
    if len(selection) > 40:
        text_selection.SetTextSize(0.025)
    if len(selection) > 50:
        text_selection.SetTextSize(0.015)

    text_selection.SetTextFont(42)
    text_selection.DrawLatexNDC(0.14, 0.82, selection)

    text_lumi = ROOT.TLatex()
    text_lumi.SetTextSize(0.04)
    text_lumi.SetTextFont(42)
    text_lumi.DrawLatexNDC(0.14, 0.77, "L = "+str(intLumi)+" pb^-1")
   
    text_e_cm = ROOT.TLatex()
    text_e_cm.SetTextSize(0.04)
    text_e_cm.SetTextFont(42)
    text_e_cm.DrawLatexNDC(0.14, 0.72, "#sqrt{s} = 91 GeV")
    """
    #Significances print
    text_title_significance = ROOT.TLatex()
    text_title_significance.SetTextSize(0.02)
    text_title_significance.SetTextFont(42)
    text_title_significance.DrawLatexNDC(0.72, 0.43, "s/#sqrt{s+b} | Z(#sigma_{" +str(estimated_Err*100) +"%})")

    text_50GeV_significance = ROOT.TLatex()
    text_50GeV_significance.SetTextSize(0.02)
    text_50GeV_significance.SetTextFont(42)
    text_50GeV_significance.DrawLatexNDC(0.65, 0.35, "50 GeV :" + str(round(significance_50GeV, 4))+ "| Z = " + str(round(Z_50GeV, 4)))

    text_20GeV_significance = ROOT.TLatex()
    text_20GeV_significance.SetTextSize(0.02)
    text_20GeV_significance.SetTextFont(42)
    text_20GeV_significance.DrawLatexNDC(0.65, 0.4, "20 GeV :" + str(round(significance_20GeV, 4)) + "| Z = " + str(round(Z_20GeV, 4)))

    text_70GeV_significance = ROOT.TLatex()
    text_70GeV_significance.SetTextSize(0.02)
    text_70GeV_significance.SetTextFont(42)
    text_70GeV_significance.DrawLatexNDC(0.65, 0.3, "70 GeV :" + str(round(significance_70GeV, 4))+ "| Z = " + str(round(Z_70GeV, 4)))
    """

    pad1.RedrawAxis()
    
    #Save plot
    c.SaveAs(output_dir_sel + "RecoDiJetElectron_invMass" + "_" + selection + ".png") if log_scale == False else c.SaveAs(output_dir_sel + "log_RecoDiJetElectron_invMass_" + selection + ".png")
    return

def get_n_events(stage1_path):
    '''
    Get the total number of events for all the files contained in stage1_path directory
    '''
    #file_pattern = stage1_path +"/chunk*.root"
    file_list = os.listdir(stage1_path)

    n_entries = 0

    for file_name in file_list:
        file = ROOT.TFile.Open(stage1_path + file_name, "READ")
        tree = file.Get("events")
        n_entries+=tree.GetEntries()
    return n_entries


def make_hist_bg(inputDir, files_list, var):
    '''
    This function makes a list of histograms from all the files contained in a directory, for a given variable
    inputDir : path to directory containing the root files for a given selection
    files_list : list of files contained in inputDir
    var : The variable of the histogram to make
    '''
    h_list = []
    for f in files_list:
        print("Looking at file", f)
        my_file = ROOT.TFile.Open(inputDir + f)
        print("Getting histogram for variable", var)
        hist = my_file.Get(var)
        hist.SetDirectory(0)
        h_list.append(hist)
        print("Histogram added to h_list")
        my_file.Close()
        print("-----------------------")
    return h_list

def merge_bg(stage1_path, input_bg_file, variable, xsec):
    '''
    This function merges the histograms of different .ROOT files for a given variable.
    input_bg_file : path to the directory containing the root files of a given selection (ChunkX_selection.root) 
    variable : the variable of the histogram to merge
    xsec : cross-section of the given process
    '''
    #Get a list containing all the root files contained in the directory : [chunk0_selNone.root, chunk1_selNone.root, etc..]
    bg_files_list = os.listdir(input_bg_file)
    
    #Get the histogram for each of the files (for a given variable) an put it in h_list.
    h_list = make_hist_bg(input_bg_file, bg_files_list, variable)

    #Loop over the histograms in h_lists and merge them.
    norm_list=[]
    for ihist, hist in enumerate(h_list):
        if ihist == 0:                            #For the first entry, we clone the first histogram from the list
            merged_hist = hist.Clone()
        else :
            merged_hist.Add(hist)                 #For the next histograms, we merge them using hist.Add().
    
    #Get the number of events of the final histogram
    n_tot = get_n_events(stage1_path)
    n_events = merged_hist.GetEntries() #Compare n_selection with n_total
    print("Total number of events", n_tot)
    print("N_events:", n_events) 
    #Scale histograms
    if merged_hist.Integral() > 0:
        merged_hist.Scale(n_events/merged_hist.Integral())       #Scale histogram to n_events
    else:
        merged_hist.Scale(n_events)
    print("Scale to number of events:", merged_hist.Integral())      #Print integral to check what is going on
    x = (intLumi * xsec)/n_tot
    print("Scale factor (L_int*N)/xsec =", x)
    merged_hist.Scale(x)
    print("Scale to x:", merged_hist.Integral())
    
    return merged_hist, n_events

df = pd.DataFrame(columns=['Selection', '20GeV', '50GeV', '70GeV', '4-body', 'Zcc', 'Zbb'])


for selection in selection_list:
    print("------------------------------------------------------")
    print("Starting selection :", selection)

    output_dir_sel =  output_dir + "Mass_comp/" + selection +'/'

    if not os.path.exists(output_dir_sel):
        os.mkdir(output_dir_sel)
        print("Directory ",output_dir_sel," Created ")
    else:
        print("Directory ",output_dir_sel," already exists")

    files_list_signal = []
    files_list_bg = []

    #Get the path to ChunkX.root files
    stage1_4body = "/eos/user/d/dimoulin/analysis_outputs/4body_W2023/output_stage1/"
    stage1_20GeV = "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_20GeV_1e-3Ve_W2023/output_stage1/"
    stage1_50GeV = "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_50GeV_1e-3Ve_W2023/output_stage1/"
    stage1_70GeV = "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_stage1/"

    stage1_Zcc = "/eos/user/d/dimoulin/analysis_outputs/HTCondor/output_stage1/p8_ee_Zcc_ecm91/"
    stage1_Zbb = "/eos/user/d/dimoulin/analysis_outputs/HTCondor/output_stage1/p8_ee_Zbb_ecm91/" 

    #Get path to selected_histograms for Zbb, Zcc
    input_Zcc = "/eos/user/d/dimoulin/plotting/selected_hist_backgrounds/p8_ee_Zcc_ecm91/" + selection + "/"
    input_Zbb = "/eos/user/d/dimoulin/plotting/selected_hist_backgrounds/p8_ee_Zbb_ecm91/" + selection + "/"

    #Get path for 4body, and HNLs
    file_4body = [input_dir + '4body_W2023_' + selection + '.root', "4-body", 0.0100, stage1_4body]

    file_HNL_50GeV = [input_dir + 'HNL_Dirac_ejj_50GeV_1e-3Ve_W2023_'+selection+'.root', "50 GeV", 0.00226, stage1_50GeV]
    file_HNL_20GeV = [input_dir + 'HNL_Dirac_ejj_20GeV_1e-3Ve_W2023_'+selection+'.root', "20 GeV", 0.00376, stage1_20GeV]
    file_HNL_70GeV = [input_dir + 'HNL_Dirac_ejj_70GeV_1e-3Ve_W2023_'+selection+'.root', "70 GeV", 0.000903, stage1_70GeV]

    #Append signal files to list
    files_list_signal.append(file_HNL_20GeV)
    files_list_signal.append(file_HNL_50GeV)
    files_list_signal.append(file_HNL_70GeV)

    #Append bg files to list
    files_list_bg.append(file_4body)

    #Make legends
    legend_list_bg = []
    legend_list_bg.append("4-body")
    legend_list_bg.append("Zcc")
    legend_list_bg.append("Zbb")

    legend_list_signal = [f[1] for f in files_list_signal]

    #Create hist lists and recover n_events for each selection:
    h_list_signal, list_n_events_signal = make_hist(files_list_signal)
    h_list_bg, list_n_events_bg = make_hist(files_list_bg)
    #Merge Zcc and Zbb histograms
    Zcc_hist, n_events_Zcc = merge_bg(stage1_Zcc, input_Zcc, "RecoDiJetElectron_invMass", 5215.46)
    Zbb_hist, n_events_Zbb = merge_bg(stage1_Zbb, input_Zbb, "RecoDiJetElectron_invMass", 6645.46)
    
    #Append merged histograms to hist_bg_list
    h_list_bg.append(Zcc_hist)
    h_list_bg.append(Zbb_hist)

    #Append n_events to bg_list
    list_n_events_bg.append(n_events_Zcc)
    list_n_events_bg.append(n_events_Zbb)

    list_n_events = list_n_events_signal + list_n_events_bg

    #Append number of event list to df with the selection:
    n_event_selection = [selection] + list_n_events
    df.loc[len(df)] = n_event_selection
  
    #Create plots
    make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, selection)

df.to_csv('Test_Generalized_Bg/Mass_comp/n_events.txt', sep='\t', index=True)
print(df)
