import ROOT
from ROOT import *
import numpy as np
import math
import os

estimated_Err = 0.1 #estimated error on significance

log_scale = True
variable = 'RecoLeadJet_e'
norm_hist = True #Normalize signal/background to 1

input_dir_bkg = "/eos/user/t/tcritchl/outputs/output_final/testingAll/"
input_dir_sgl = "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/HNL_sample_creation/signal_HNLS/out_final/"
output_dir =  "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/HNL_sample_creation/signal_HNLS/SignalvsBackground/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

selection_list = [
   "selNone",
   #"selMissingEGt10",
   #"selEleEGt13_MissingEGt10",
   #"selEleEGt13_MissingEGt10_EleSecondJetDRGt06",
   #"selEleEGt13_MissingEGt10_EleSecondJetDRGt06_EleLeadJetDRGt06",
   #"selEleEGt13_MissingEGt10_EleSecondJetDRGt06_EleLeadJetDRGt06Lt32",
]

colors_signal = [876, 616, 880, 801, 629, 879, 602, 921, 622]
colors_bg = [856, 410, 801, 629, 879, 602, 921, 622]


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


def get_max_Z(h_signal, h_list_bg, x1, x2):
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

def get_max_significance(h_signal, h_list_bg, x1, x2):
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
            significance = s/math.sqrt(s+b)
        h_significance.SetBinContent(bin, significance)

        if h_significance.GetBinContent(bin) > h_significance_max:
            h_significance_max = h_significance.GetBinContent(bin)
    return h_significance_max


def make_hist(files_list):
	h_list = []
	for f in files_list:
		print("Looking at file", f[1])
		my_file = ROOT.TFile.Open(f[0])
		print("Getting histogram for variable", f[1])
		hist = my_file.Get("RecoDiJetElectron_invMass")
		hist.SetDirectory(0)
		h_list.append(hist)
		print("Histogram added to h_list")
		my_file.Close()
		print("-----------------------")
	return h_list

def make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, selection):
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
    
    #Compute significances 
    significance_20GeV = get_max_significance(h_list_signal[0], h_list_bg, 15, 25)
    significance_50GeV = get_max_significance(h_list_signal[1], h_list_bg, 45, 55)
    significance_70GeV = get_max_significance(h_list_signal[2], h_list_bg, 65, 75)

    #Compute Z
    Z_20GeV = get_max_Z(h_list_signal[0], h_list_bg, 15, 25)
    Z_50GeV = get_max_Z(h_list_signal[1], h_list_bg, 45, 55)
    Z_70GeV = get_max_Z(h_list_signal[2], h_list_bg, 65, 75)
 
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
    for ih,h in enumerate(h_list_signal):
        h.SetLineColor(colors_signal[ih])
        #h.SetFillColorAlpha(colors_signal[ih], 0.3)
        h.SetLineWidth(3)
        h.SetStats(0)
        h.GetXaxis().SetTitle("m [GeV]")
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
        h.GetXaxis().SetTitle("m [GeV]")
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
    if len(selection) > 40:
        text_selection.SetTextSize(0.025)
    if len(selection) > 50:
        text_selection.SetTextSize(0.015)

    text_selection.SetTextFont(42)
    text_selection.DrawLatexNDC(0.14, 0.82, selection)

    text_lumi = ROOT.TLatex()
    text_lumi.SetTextSize(0.04)
    text_lumi.SetTextFont(42)
    text_lumi.DrawLatexNDC(0.14, 0.77, "L = 150 ab^{-1}")
   
    text_e_cm = ROOT.TLatex()
    text_e_cm.SetTextSize(0.04)
    text_e_cm.SetTextFont(42)
    text_e_cm.DrawLatexNDC(0.14, 0.72, "#sqrt{s} = 91 GeV")

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

 
    pad1.RedrawAxis()

    c.SaveAs(output_dir_sel + files_list_signal[0][1] + "_" + selection + ".png") if log_scale == False else c.SaveAs(output_dir_sel + "log_RecoDiJetElectron_invMass_" + selection + ".png")
    return

for selection in selection_list:
    output_dir_sel = output_dir

    if not os.path.exists(output_dir_sel):
        os.mkdir(output_dir_sel)
        print("Directory ",output_dir_sel," Created ")
    else:
        print("Directory ",output_dir_sel," already exists")

    files_list_signal = []
    files_list_bg = []

    #file_4body = [input_dir + '4body_W2023_' + selection + '.root', "4-body"]
    file_Zud = [input_dir_bkg + 'p8_ee_Zud_ecm91_'+selection+'_histo'+'.root',variable, "Z -> ud"]
    file_Zbb = [input_dir_bkg + 'p8_ee_Zbb_ecm91_'+selection+'_histo'+'.root',variable, "Z -> bb"]
    file_Zcc = [input_dir_bkg + 'p8_ee_Zcc_ecm91_'+selection+'_histo'+'.root',variable, "Z -> cc"]

    file_HNL_50GeV = [input_dir_sgl + 'HNL_ejj_50GeV_'+selection+'_histo'+'.root',variable, "50 GeV"]
    file_HNL_20GeV = [input_dir_sgl + 'HNL_ejj_20GeV_'+selection+'_histo'+'.root',variable, "20 GeV"]
    file_HNL_70GeV = [input_dir_sgl + 'HNL_ejj_70GeV_'+selection+'_histo'+'.root',variable,"70 GeV"]

    files_list_signal.append(file_HNL_20GeV)
    files_list_signal.append(file_HNL_50GeV)
    files_list_signal.append(file_HNL_70GeV)

    files_list_bg.append(file_Zbb)
    files_list_bg.append(file_Zcc)
    files_list_bg.append(file_Zud)
    #files_list_bg.append(file_4body)

    legend_list_bg = [f[1] for f in files_list_bg]
    legend_list_signal = [f[1] for f in files_list_signal]

    h_list_signal = make_hist(files_list_signal)
    h_list_bg = make_hist(files_list_bg)   

    make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, selection)



