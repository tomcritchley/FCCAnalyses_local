import ROOT
from ROOT import *
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import itertools

intLumi = 150.0e+06 #fb-1 (150 ab-1)

log_scale = False 

norm_hist = True #Normalize signal/background to 1

input_dir = "selected_hist/"
output_dir =  "Test_Generalized_Bg/"


if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

selection_list = [
  "selNone",
#  "selMissingEGt10",
#  "selMissingEGt10_EleEGt13",
#  "selMissingEGt10_EleEGt25",
#  "selMissingEGt10_EleEGt30",
#  "selMissingEGt10_EleEGt35",
]

variables_list = [
#   "RecoDiJet_delta_R",
   "RecoElectron_lead_e",
#   "RecoElectron_LeadJet_delta_R",
#   "RecoElectron_SecondJet_delta_R",
#   "RecoElectron_DiJet_delta_R",
#   "RecoMissingEnergy_e",
#   "RecoDiJetElectron_invMass"
]

colors_signal = [876, 616, 880, 801, 629, 879, 602, 921, 622]
colors_bg = [801, 410, 856, 629, 879, 602, 921, 622]


def get_z(n,b,e,b_thr=0):
  '''
  n: total number of events
  b: prediction of bkg
  e: error on the prediction
  '''
  ss=e*e  # sigma squared
  bb=b*b
  print("bb =", bb)
  print("ss =", ss)

  if n<= 0 or b<= b_thr:
    z=0
  elif b==0:
    z=0
  else:
    arg =  2*( n*np.log((n*b+n*ss)/(bb+n*ss)) - ((bb)/ss)*np.log(1+(ss*n-ss*b)/(bb+b*ss)) )
    if arg <0:
        z=0
        print("Arg from sqrt is negative")
    else:
        z=np.sqrt( 2*( n*np.log((n*b+n*ss)/(bb+n*ss)) - ((bb)/ss)*np.log(1+(ss*n-ss*b)/(bb+b*ss)) ))
        print("Arg from sqrt=",arg) 
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

		hist.SetDirectory(0)
		h_list.append(hist)
		print("Histogram added to h_list")
		my_file.Close()
		print("-----------------------")
	return h_list

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
        h_Z_integral = h_Z.Integral()
        print("H_Z integral:",h_Z_integral, " Signal integral():", h.Integral())
        for bin in range(1, h_Z.GetNbinsX()+1):
            s = h_Z.Integral(bin, h.GetNbinsX())
            b = h_list_bg[0].Integral(bin, h_list_bg[0].GetNbinsX()) + h_list_bg[1].Integral(bin, h_list_bg[1].GetNbinsX()) + h_list_bg[2].Integral(bin, h_list_bg[2].GetNbinsX())
            Z = 0
            print("#####################################")
            print("## bin :", bin)
            print("## signal s=",s)
            print("## background b=",b)
            n = s+b
            if n > 0:
                Z = get_z(n,b, 0.1*b , 0)
            elif n < 0:
                print("Number of events for Z computation is negative: n=", n)
            else:
                Z = 0
            print("## Z =", Z)
        h_Z.SetBinContent(bin, Z)
        if h_Z.GetBinContent(bin) > h_Z_max:      #Set max to the highest bin value
            h_Z_max = h_Z.GetBinContent(bin)
        h_Z.Scale(1/h_Z.Integral())               #Scale significance histogram to 1
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


def make_hist_bg(inputDir, files_list, var):
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

def merge_bg(input_bg_file, variable, xsec, stage1_path):
    bg_files_list = os.listdir(input_bg_file)

    h_list = make_hist_bg(input_bg_file, bg_files_list, variable)
    for ihist, hist in enumerate(h_list):
        if ihist == 0:
            merged_hist = hist.Clone("merged_hist")
        else :
            merged_hist.Add(hist)

    n_tot = get_n_events(stage1_path)
    n_events = merged_hist.GetEntries()
    merged_hist.Scale(n_events/merged_hist.Integral())   #Scale histogram to number of events
    print("Scale to n:", merged_hist.Integral())
    x = (intLumi*xsec)/n_tot                             #Compute Scale factor x=L*xsec/N
    print("Scaling factor (L_int*N)/xsec =", x)
    merged_hist.Scale(x)                                 
    print("hist integral:", merged_hist.Integral())
    return merged_hist

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

stage1_4body = "/eos/user/d/dimoulin/analysis_outputs/4body_W2023/output_stage1/"
stage1_20GeV = "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_20GeV_1e-3Ve_W2023/output_stage1/"
stage1_50GeV = "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_50GeV_1e-3Ve_W2023/output_stage1/"
stage1_70GeV = "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_stage1/"

stage1_Zcc = "/eos/user/d/dimoulin/analysis_outputs/HTCondor/output_stage1/p8_ee_Zcc_ecm91/"
stage1_Zbb = "/eos/user/d/dimoulin/analysis_outputs/HTCondor/output_stage1/p8_ee_Zbb_ecm91/"


for selection in selection_list:
    output_dir_sel = output_dir + "/Variables_comp/" + selection +'/'

    if not os.path.exists(output_dir_sel):
        os.mkdir(output_dir_sel)
        print("Directory ",output_dir_sel," Created ")
    else:
        print("Directory ",output_dir_sel," already exists")
    
    files_list_signal = []
    files_list_bg = []

    input_Zcc = "/eos/user/d/dimoulin/plotting/selected_hist_backgrounds/p8_ee_Zcc_ecm91/" + selection + "/"
    input_Zbb = "/eos/user/d/dimoulin/plotting/selected_hist_backgrounds/p8_ee_Zbb_ecm91/" + selection + "/"    

    file_4body = [input_dir + '4body_W2023_' + selection + '.root', "4-body", 0.0103, stage1_4body]

    file_HNL_50GeV = [input_dir + 'HNL_Dirac_ejj_50GeV_1e-3Ve_W2023_'+selection+'.root', "50 GeV", 0.00226, stage1_50GeV]
    file_HNL_20GeV = [input_dir + 'HNL_Dirac_ejj_20GeV_1e-3Ve_W2023_'+selection+'.root', "20 GeV", 0.00376, stage1_20GeV]
    file_HNL_70GeV = [input_dir + 'HNL_Dirac_ejj_70GeV_1e-3Ve_W2023_'+selection+'.root', "70 GeV", 0.000903, stage1_70GeV]

    files_list_signal.append(file_HNL_20GeV)
    files_list_signal.append(file_HNL_50GeV)
    files_list_signal.append(file_HNL_70GeV)

    files_list_bg.append(file_4body)

    legend_list_bg = []
    legend_list_bg.append("4-body")
    legend_list_bg.append("Zcc")
    legend_list_bg.append("Zbb")

    legend_list_signal = [f[1] for f in files_list_signal]

    for variable in variables_list:
        h_list_signal = make_hist(files_list_signal, variable)
        h_list_bg = make_hist(files_list_bg, variable)

        Zcc_hist = merge_bg(input_Zcc, variable, 5215.46, stage1_Zcc)
        Zbb_hist = merge_bg(input_Zbb, variable, 6645.46, stage1_Zbb)
        h_list_bg.append(Zcc_hist)
        h_list_bg.append(Zbb_hist)
        make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, variable)

