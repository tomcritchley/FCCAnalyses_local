import ROOT
from ROOT import *
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import itertools

selection = "selNone"
#selection = "selRecoEleGt0"
#selection = "selRecoEleEGt11"
#selection = "selMissingEGt25"
#selection = "selEleDiJetDRLt27"
#selection = "selEleSecondJetDRGt2"
#selection = "selEleEGt11_MissingEGt25"
#selection = "selEleEGt11_MissingEGt25_EleDiJetDRLt27"
#selection = "selEleEGt11_MissingEGt25_EleDiJetDRLt27_EleSecondJetDRGt2"

log_scale = True

norm_hist = False #Normalize signal/background to 1

input_dir = "selected_hist/"
output_dir =  "SignalVsBackground/"
output_dir_sel = "SignalVsBackground/Mass_comp/" + selection +'/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

if not os.path.exists(output_dir_sel):
    os.mkdir(output_dir_sel)
    print("Directory ",output_dir_sel," Created ")
else:
    print("Directory ",output_dir_sel," already exists")

file_4body = input_dir + '4body_W2023_' + selection + '.root'
file_Zbb = input_dir + 'p8_ee_Zbb_ecm91_W2023_' + selection + '.root'
file_Zcc = input_dir + 'p8_ee_Zcc_ecm91_W2023_' + selection + '.root'

file_HNL_50GeV = input_dir + 'HNL_Dirac_ejj_50GeV_1e-3Ve_W2023_'+selection+'.root' 
file_HNL_20GeV = input_dir + 'HNL_Dirac_ejj_20GeV_1e-3Ve_W2023_'+selection+'.root'
file_HNL_70GeV = input_dir + 'HNL_Dirac_ejj_70GeV_1e-3Ve_W2023_'+selection+'.root'


files_list_signal = [
   [file_HNL_20GeV, "RecoDiJetElectron_invMass", "20GeV HNL"],
   [file_HNL_50GeV, "RecoDiJetElectron_invMass", "50GeV HNL"],
   [file_HNL_70GeV, "RecoDiJetElectron_invMass", "70GeV HNL"],


]
files_list_bg = [
   [file_Zbb, "RecoDiJetElectron_invMass", "Z -> bb"],
   [file_Zcc, "RecoDiJetElectron_invMass", "Z -> cc"],
   [file_4body, "RecoDiJetElectron_invMass", "4 body"]
]

legend_list_bg = [f[2] for f in files_list_bg]
ratio_list_bg = [f[2] for f in files_list_bg]

legend_list_signal = [f[2] for f in files_list_signal]
ratio_list_signal = [f[2] for f in files_list_signal]

colors_signal = [876, 616, 880, 801, 629, 879, 602, 921, 622]
colors_bg = [856, 410, 801, 629, 879, 602, 921, 622]


def make_hist(files_list):
	h_list = []
	for f in files_list:
		print("Looking at file", f[2])
		my_file = ROOT.TFile.Open(f[0])
		print("Getting histogram for variable", f[1])
		hist = my_file.Get(f[1])
		hist.SetDirectory(0)
		h_list.append(hist)
		print("Histogram added to h_list")
		my_file.Close()
		print("-----------------------")
	return h_list

h_list_signal = make_hist(files_list_signal)
h_list_bg = make_hist(files_list_bg)

def make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg):
   #  print('looking at histogram:', plot_info[0])
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
    
    #Compute max
    h_list = h_list_signal + h_list_bg
    h_max = 0
    for ih,h in enumerate(h_list):
        if h.GetMaximum() > h_max:
            h_max = h.GetMaximum()
    for ih,h in enumerate(h_list_signal):
        leg_sig.AddEntry(h, legend_list_signal[ih])
    for ih,h in enumerate(h_list_bg):
        leg_bg.AddEntry(h, legend_list_bg[ih])

    #Build significance and get the max for each signal
    h_significance_max_list = []

    for ih, h in enumerate(h_list_signal):
        h_significance = h.Clone("h_significance")
        h_significance_max = 0
        for bin in range(1, h_significance.GetNbinsX()+1):
            s = h.Integral(bin, h.GetNbinsX())
            b = h_list_bg[0].Integral(bin, h_list_bg[0].GetNbinsX()) + h_list_bg[1].Integral(bin, h_list_bg[1].GetNbinsX()) + h_list_bg[2].Integral(bin, h_list_bg[2].GetNbinsX())
            significance = 0
            if s+b > 0:
                significance = s / ROOT.TMath.Sqrt(s + b)
            h_significance.SetBinContent(bin, significance)
    
            if h_significance.GetBinContent(bin) > h_significance_max:
                h_significance_max = h_significance.GetBinContent(bin)
        h_significance_max_list.append(h_significance_max)            
    
    # Draw in the top panel
    pad1.cd()
    for ih,h in enumerate(h_list_signal):
        h.SetLineColor(colors_signal[ih])
        #h.SetFillColorAlpha(colors_signal[ih], 0.3)
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle(files_list_signal[0][1])
        h.GetYaxis().SetTitle("Entries") if log_scale == False else h.GetYaxis().SetTitle("log Entries")
        h.GetYaxis().SetTitleOffset(0.8)
        h.SetMaximum(1.25*h_max)
        if norm_hist == True :
            h.Scale(1.0 / h.Integral())
            h_max = 0
            if h.GetMaximum() > h_max:
                h_max = h.GetMaximum()
            h.SetMaximum(h_max*1.25)
        h.Draw('hist same')

    for ih,h in enumerate(h_list_bg):
        h.SetLineColor(colors_bg[ih])
        #h.SetFillColorAlpha(colors_bg[ih], 0.3)
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle(files_list_signal[0][1])
        h.GetYaxis().SetTitle("Entries") if log_scale == False else h.GetYaxis().SetTitle("log Entries")
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
    text_selection.SetTextFont(42)
    text_selection.DrawLatexNDC(0.14, 0.82, selection)

    text_lumi = ROOT.TLatex()
    text_lumi.SetTextSize(0.04)
    text_lumi.SetTextFont(42)
    text_lumi.DrawLatexNDC(0.14, 0.77, "L = 150 ab^-1")
   
    text_e_cm = ROOT.TLatex()
    text_e_cm.SetTextSize(0.04)
    text_e_cm.SetTextFont(42)
    text_e_cm.DrawLatexNDC(0.14, 0.72, "#sqrt{s} = 91 GeV")

    #Significances print
    text_50GeV_significance = ROOT.TLatex()
    text_50GeV_significance.SetTextSize(0.02)
    text_50GeV_significance.SetTextFont(42)
    text_50GeV_significance.DrawLatexNDC(0.7, 0.35, "50 GeV :" + str(round(h_significance_max_list[1], 4)))

    text_20GeV_significance = ROOT.TLatex()
    text_20GeV_significance.SetTextSize(0.02)
    text_20GeV_significance.SetTextFont(42)
    text_20GeV_significance.DrawLatexNDC(0.7, 0.4, "20 GeV :" + str(round(h_significance_max_list[0], 4)))

    text_70GeV_significance = ROOT.TLatex()
    text_70GeV_significance.SetTextSize(0.02)
    text_70GeV_significance.SetTextFont(42)
    text_70GeV_significance.DrawLatexNDC(0.7, 0.3, "70 GeV :" + str(round(h_significance_max_list[2], 4)))

 
    pad1.RedrawAxis()

    c.SaveAs(output_dir_sel + files_list_signal[0][1] + "_" + selection + ".png") if log_scale == False else c.SaveAs(output_dir_sel + "log_" + files_list_signal[0][1] +"_" + selection + ".png")
    return


make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg)



