import ROOT
from ROOT import *
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import itertools

log_scale = False 

norm_hist = True #Normalize signal/background to 1

input_dir = "selected_hist/"
output_dir =  "SignalVsBackground/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

selection_list = [
   "selNone",
   "selMissingEGt10",
   "selEleEGt13_MissingEGt10",
   "selEleEGt13_MissingEGt10_EleSecondJetDRGt06",
   "selEleEGt13_MissingEGt10_EleSecondJetDRGt06_EleLeadJetDRGt06",
]

variables_list = [
   "RecoDiJet_delta_R",
   "RecoElectron_lead_e",
   "RecoElectron_LeadJet_delta_R",
   "RecoElectron_SecondJet_delta_R",
   "RecoElectron_DiJet_delta_R",
   "RecoMissingEnergy_e",
]

colors_signal = [876, 616, 880, 801, 629, 879, 602, 921, 622]
colors_bg = [856, 410, 801, 629, 879, 602, 921, 622]


def get_z(n,b,e,b_thr=0):
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

def make_hist(files_list, var):
	h_list = []
	for f in files_list:
		print("Looking at file", f[1])
		my_file = ROOT.TFile.Open(f[0])
		print("Getting histogram for variable", var)
		hist = my_file.Get(var)
		hist.SetDirectory(0)
		h_list.append(hist)
		print("Histogram added to h_list")
		my_file.Close()
		print("-----------------------")
	return h_list

#h_list_signal = make_hist(files_list_signal, "RecoDiJet_delta_R")
#h_list_bg = make_hist(files_list_bg, "RecoDiJet_delta_R")

def make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, variable):
   #  print('looking at histogram:', plot_info[0])
    c = ROOT.TCanvas("can","can",600,600)
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1)
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0, 1, 0.3)

    pad1.SetBottomMargin(0.035)
    if log_scale == True : pad1.SetLogy()
    pad1.SetTickx()
    pad1.SetTicky()
    pad1.Draw()

    pad2.SetTopMargin(0.03)
    pad2.SetBottomMargin(0.25)
    pad2.Draw()


    leg_bg = ROOT.TLegend(0.7, 0.5, 0.9, 0.65)
    leg_bg.SetFillStyle(0)
    leg_bg.SetLineWidth(0)
    
    leg_sig = ROOT.TLegend(0.7, 0.70, 0.9, 0.85)
    leg_sig.SetFillStyle(0)
    leg_sig.SetLineWidth(0)

    h_list = h_list_signal + h_list_bg

    h_max = 0
    for ih,h in enumerate(h_list):
        if h.GetMaximum() > h_max:
            h_max = h.GetMaximum()
    for ih,h in enumerate(h_list_signal):
        leg_sig.AddEntry(h, legend_list_signal[ih])
    for ih,h in enumerate(h_list_bg):
        leg_bg.AddEntry(h, legend_list_bg[ih])

    pad2.cd()

    h_Z_list = []
    h_Z_max = 0
    for ih, h in enumerate(h_list_signal):
        h_Z = h.Clone("h_Z")

        for bin in range(1, h_Z.GetNbinsX()+1):
            s = h.Integral(bin, h.GetNbinsX())
            b = h_list_bg[0].Integral(bin, h_list_bg[0].GetNbinsX()) + h_list_bg[1].Integral(bin, h_list_bg[1].GetNbinsX()) + h_list_bg[2].Integral(bin, h_list_bg[2].GetNbinsX())
            Z = 0
            n = s+b
            if n > 0:
                Z = get_z(n,b, 0.1*b , 0)
            h_Z.SetBinContent(bin, Z)
            if h_Z.GetBinContent(bin) > h_Z_max:
                h_Z_max = h_Z.GetBinContent(bin)
        h_Z_list.append(h_Z.Clone("h_Z_"+str(ih)))

    for ih, h in enumerate(h_Z_list):
        h.SetLineColor(colors_signal[ih])
        h.SetMinimum(0)
        h.SetStats(0)
        h.SetMaximum(h_Z_max*1.25)
        h.GetYaxis().SetLabelSize(h.GetYaxis().GetLabelSize()*1.6)
        h.GetYaxis().SetLabelOffset(0.01)
        h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*2.5)
        h.GetYaxis().SetTitleOffset(0.45)
        h.GetYaxis().SetTitle("Z(#sigma_{10%})")
        h.GetXaxis().SetTitle(variable)
        h.GetXaxis().SetLabelSize(h.GetXaxis().GetLabelSize()*2.3)
        h.GetXaxis().SetLabelOffset(0.02)
        h.GetXaxis().SetTitleSize(h.GetXaxis().GetTitleSize()*3)
        h.GetXaxis().SetTitleOffset(1.05)

        h.Draw("hist same")

    # Draw in the top panel
    pad1.cd()
    for ih,h in enumerate(h_list_signal):
        h.SetLineColor(colors_signal[ih])
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle(variable)
        h.GetYaxis().SetTitle("Entries") if log_scale == False else h.GetYaxis().SetTitle("log Entries")
        h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*1.5)
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
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle(variable)
        h.GetYaxis().SetTitle("Entries") if log_scale == False else h.GetYaxis().SetTitle("log Entries")
        h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*1.5)
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
     
   # c.SaveAs("test.png")
    c.SaveAs(output_dir_sel + variable + "_" + selection + ".png") if log_scale == False else c.SaveAs(output_dir_sel + "log_" + variable +"_" + selection + ".png")
    return


for selection in selection_list:
    output_dir_sel = "SignalVsBackground/Variables_comp/" + selection +'/'

    if not os.path.exists(output_dir_sel):
        os.mkdir(output_dir_sel)
        print("Directory ",output_dir_sel," Created ")
    else:
        print("Directory ",output_dir_sel," already exists")
    
    files_list_signal = []
    files_list_bg = []

    file_4body = [input_dir + '4body_W2023_' + selection + '.root', "4-body"]
    file_Zbb = [input_dir + 'p8_ee_Zbb_ecm91_W2023_' + selection + '.root', "Z -> bb"]
    file_Zcc = [input_dir + 'p8_ee_Zcc_ecm91_W2023_' + selection + '.root', "Z -> cc"]

    file_HNL_50GeV = [input_dir + 'HNL_Dirac_ejj_50GeV_1e-3Ve_W2023_'+selection+'.root', "50 GeV"]
    file_HNL_20GeV = [input_dir + 'HNL_Dirac_ejj_20GeV_1e-3Ve_W2023_'+selection+'.root', "20 GeV"]
    file_HNL_70GeV = [input_dir + 'HNL_Dirac_ejj_70GeV_1e-3Ve_W2023_'+selection+'.root', "70 GeV"]

    files_list_signal.append(file_HNL_20GeV)
    files_list_signal.append(file_HNL_50GeV)
    files_list_signal.append(file_HNL_70GeV)

    files_list_bg.append(file_Zbb)
    files_list_bg.append(file_Zcc)
    files_list_bg.append(file_4body)

    legend_list_bg = [f[1] for f in files_list_bg]
    legend_list_signal = [f[1] for f in files_list_signal]

    for variable in variables_list:
        h_list_signal = make_hist(files_list_signal, variable)
        h_list_bg = make_hist(files_list_bg, variable)
        make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, variable)



