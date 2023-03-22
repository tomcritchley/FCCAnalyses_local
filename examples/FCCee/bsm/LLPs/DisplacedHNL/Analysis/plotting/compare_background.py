import ROOT
from ROOT import *
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import itertools


HNL_mass = "50GeV"

#selection = "selNone"
selection = "selRecoEleGt0"

log_scale = True

input_dir = "selected_hist/"
output_dir =  HNL_mass + "_SignalVsBackground/"
output_dir_sel = HNL_mass + "_SignalVsBackground/" + selection +'/'

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
file_HNL = input_dir + 'HNL_Dirac_ejj_'+HNL_mass+'_1e-3Ve_W2023_'+selection+'.root' 

files_list_signal = [
   [file_HNL, "RecoDiJetElectron_invMass", "50GeV HNL"]
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

colors_signal = [609, 856, 410, 801, 629, 879, 602, 921, 622]
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
    pad1 = ROOT.TPad("pad1", "pad1",0.0,0.35,1.0,1.0,21)
    pad2 = ROOT.TPad("pad2", "pad2",0.0,0.0,1.0,0.35,22)

    pad1.SetFillColor(0)
    pad1.SetBottomMargin(0.01)
    if log_scale == True : pad1.SetLogy()
    pad1.SetTickx()
    pad1.SetTicky()
    pad1.Draw()

    pad2.SetFillColor(0)
    pad2.SetTopMargin(0.01)
    pad2.SetBottomMargin(0.3)
    pad2.Draw()

    leg_bg = ROOT.TLegend(0.7, 0.5, 0.9, 0.65)
    leg_bg.SetFillStyle(0)
    leg_bg.SetLineWidth(0)
    
    leg_sig = ROOT.TLegend(0.12, 0.5, 0.32, 0.65)
    leg_sig.SetFillStyle(0)
    leg_sig.SetLineWidth(0)

    h_list = h_list_signal + h_list_bg

    h_max = 0
    for ih,h in enumerate(h_list):
        if h.GetMaximum() > h_max:
            h_max = h.GetMaximum()
        h.Sumw2()
    for ih,h in enumerate(h_list_signal):
        leg_sig.AddEntry(h, legend_list_signal[ih])
    for ih,h in enumerate(h_list_bg):
        leg_bg.AddEntry(h, legend_list_bg[ih])

    # Draw in the top panel
    pad1.cd()
    for ih,h in enumerate(h_list_signal):
        h.SetLineColor(colors_signal[ih])
        #h.SetFillColorAlpha(colors_signal[ih], 0.3)
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle("RecoDiJetElectron_invMass")
        h.GetYaxis().SetTitle("Entries") if log_scale == False else h.GetYaxis().SetTitle("log Entries")
        h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*1.5)
        h.GetYaxis().SetTitleOffset(0.8)
        h.SetMaximum(1.25*h_max)
        h.Draw('hist same')

    for ih,h in enumerate(h_list_bg):
        h.SetLineColor(colors_bg[ih])
        #h.SetFillColorAlpha(colors_bg[ih], 0.3)
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle("RecoDiJetElectron_invMass")
        h.GetYaxis().SetTitle("Entries") if log_scale == False else h.GetYaxis().SetTitle("log Entries")
        h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*1.5)
        h.GetYaxis().SetTitleOffset(0.8)
        h.SetMaximum(1.25*h_max)
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
 
    pad1.RedrawAxis()

    # build efficiency
    
    #efficiency = h_list_signal[0].Clone("efficiency")
    #efficiency.SetStats(False)
    #efficiency.Divide(h_list_signal[0] + h_list_bg[0]+ h_list_bg[1] + h_list_bg[2])
     
    #Plot in the second pad
    #pad2.cd()
    #efficiency.SetMaximum(1)
    #efficiency.SetMinimum(0)

    #efficiency.GetYaxis().SetTitle("Efficiency")
    #efficiency.GetYaxis().SetLabelSize(efficiency.GetYaxis().GetLabelSize()*1.6)
    #efficiency.GetYaxis().SetLabelOffset(0.01)
    #efficiency.GetYaxis().SetTitleSize(efficiency.GetYaxis().GetTitleSize()*1.6)
    #efficiency.GetYaxis().SetTitleOffset(0.5)
    #efficiency.GetXaxis().SetLabelSize(efficiency.GetXaxis().GetLabelSize()*2.3)
    #efficiency.GetXaxis().SetLabelOffset(0.02)
    #efficiency.GetXaxis().SetTitleSize(efficiency.GetXaxis().GetTitleSize()*3)
    #efficiency.GetXaxis().SetTitleOffset(1.05)
    #efficiency.Draw("hist")
    
    pad2.cd()
    #Build significance
    h_significance = ROOT.TH1F("h_significance", "Significance", 50, 0, 100)

    # Fill the significance histogram with values
    for bin in range(1, h_significance.GetNbinsX()+1):
        s = h_list_signal[0].Integral(bin, h_list_signal[0].GetNbinsX())
        b = h_list_bg[0].Integral(bin, h_list_bg[0].GetNbinsX()) + h_list_bg[1].Integral(bin, h_list_bg[2].GetNbinsX()) + h_list_bg[2].Integral(bin, h_list_bg[2].GetNbinsX())
        significance = 0
        if s+b > 0:
            significance = s / ROOT.TMath.Sqrt(s + b)
        h_significance.SetBinContent(bin, significance)

    # Set the axis titles
    h_significance.SetStats(0)
    h_significance.GetYaxis().SetLabelSize(h_significance.GetYaxis().GetLabelSize()*1.6)
    h_significance.GetYaxis().SetLabelOffset(0.01)
    h_significance.GetYaxis().SetTitleSize(h_significance.GetYaxis().GetTitleSize()*1.6)
    h_significance.GetYaxis().SetTitleOffset(0.5)
    h_significance.GetXaxis().SetLabelSize(h_significance.GetXaxis().GetLabelSize()*2.3)
    h_significance.GetXaxis().SetLabelOffset(0.02)
    h_significance.GetXaxis().SetTitleSize(h_significance.GetXaxis().GetTitleSize()*3)
    h_significance.GetXaxis().SetTitleOffset(1.05) 

    # Draw the significance histogram
    h_significance.Draw()

    # build ratios
    #h_ratios = []
    #h_list = h_list_signal + h_list_bg
    #for ih,h in enumerate(h_list):
    #    if ih == 0:
    #        h_ratios.append(h.Clone('h_ratio_0'))
    #        for ibin in range(-1, h.GetNbinsX()+1):
    #            h_ratios[0].SetBinContent(ibin,1)
    #    else:
    #        h_ratios.append(h.Clone('h_ratio_'+str(ih)))
    #        h_ratios[ih].Divide(h_list[0])

    # draw in the bottom panel
    #pad2.cd()
    #for ih,h in enumerate(h_ratios):
    #    h.SetMaximum(1.5)
    #    h.SetMinimum(0.5)

    #    h.GetYaxis().SetTitle("Ratio to "+ratio_list_bg[0])
    #    h.GetYaxis().SetLabelSize(h.GetYaxis().GetLabelSize()*1.6)
    #    h.GetYaxis().SetLabelOffset(0.01)
    #    h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*1.6)
    #    h.GetYaxis().SetTitleOffset(0.5)

    #    h.GetXaxis().SetLabelSize(h.GetXaxis().GetLabelSize()*2.3)
    #    #h.GetXaxis().SetLabelOffset(0.02)
    #    h.GetXaxis().SetTitleSize(h.GetXaxis().GetTitleSize()*3)
    #    h.GetXaxis().SetTitleOffset(1.05)

    #    h.Draw('same')
    #    if ih>0:
    #        h.Draw('same')

    c.SaveAs(output_dir_sel + "SignalVsBackground_" + selection + ".png") if log_scale == False else c.SaveAs(output_dir_sel + "log_" + "SignalVsBackground_" + selection + ".png")
    return


make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg)



