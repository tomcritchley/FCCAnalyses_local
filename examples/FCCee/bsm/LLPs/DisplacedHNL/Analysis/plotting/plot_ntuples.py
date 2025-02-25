import ROOT
from ROOT import *
import numpy as np
import math
import os

ROOT.gStyle.SetExponentOffset(-0.1, 0.03, "y")

uncertainty_count_factor = 0.1  # 10% background uncertainty for the significance
variable_list = [
    ["RecoElectron_lead_e", "Reco lead electron energy [GeV]"],
    ["RecoDiJet_delta_R", "Reco di-jet #Delta R"],
    ["RecoDiJet_angle", "Reco di-jet #Psi [Rad.]"],
    ["RecoElectron_LeadJet_delta_R", "Reco lead jet #DeltaR"],
    ["RecoElectron_SecondJet_delta_R", "Reco second jet #Delta R"],
    ["RecoElectron_DiJet_delta_R", "Reco electron di-jet #Delta R"],
    ["RecoLeadElectron_Pmiss_delta_theta", "Reconstructed electron missing momentum #theta [Rad.]"],
    ["RecoElectronTrack_absD0sig", "Reco electron |d_{0}| [mm] sig"],
    ["RecoElectronTrack_absD0cov", "Reco electron |d_{0}| [mm] cov"],
    ["RecoElectronTrack_absD0", "Reco electron |d_{0}| [mm]"],
    ["RecoDiJet_phi", "Reco DiJet #phi [Rad.]"],
    ["RecoMissingEnergy_theta", "Reco Missing Energy   #theta [Rad.]"],
    ["RecoMissingEnergy_e", "Reco missing energy [GeV]"],
    ["RecoDiJetElectron_invMass", "Mass [GeV]"],
    ["ntracks", "Number of tracks"],
    ["n_primt", "Number of primary tracks"],
    ["Vertex_chi2", "Chi^{2} of the primary vertex"]
]

chosen_variable = variable_list[16]

significance_directions = ["LR", "RL"]
significance_direction = significance_directions[1]

normalisation = True
luminosity = 10000  # 10 fb^-1 as 1e4 pb^-1
log_scale = True

selection = "selNone"
input_dir_bkg = "/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/final/"  # bb cc and 4body samples
input_dir_sgl = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/final/"  # signals
output_dir = "/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/vertex/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ", output_dir, " Created ")
else:
    print("Directory ", output_dir, " already exists")

file_Zbb = input_dir_bkg + 'p8_ee_Zbb_ecm91_'+selection+'.root'
file_Zcc = input_dir_bkg + 'p8_ee_Zcc_ecm91_' + selection+'.root'
file_4body = input_dir_bkg + 'ejjnu_' + selection+'.root'

file_HNL_20 = input_dir_sgl+'HNL_Dirac_ejj_20GeV_1e-3Ve_'+selection+'.root'
file_HNL_50 = input_dir_sgl+'HNL_Dirac_ejj_50GeV_1e-3Ve_'+selection+'.root'
file_HNL_70 = input_dir_sgl+'HNL_Dirac_ejj_70GeV_1e-3Ve_'+selection+'.root'

cross_sections_sgl = [0.003771, 0.002268, 0.0009058]  # cross sections calculated by madgraph
total_events_sgl = [1, 1, 1]  # events generated by madgraph
selection_scale_sgl = [0.826038170925038/1, 1, 1]
files_list_signal = [
    [file_HNL_20, chosen_variable[0], "20GeV HNL", cross_sections_sgl[0], total_events_sgl[0], selection_scale_sgl[0]],
    [file_HNL_50, chosen_variable[0], "50GeV HNL", cross_sections_sgl[1], total_events_sgl[1], selection_scale_sgl[1]],
    [file_HNL_70, chosen_variable[0], "70GeV HNL", cross_sections_sgl[2], total_events_sgl[2], selection_scale_sgl[2]]
]

cross_sections_bg = [5215.46, 6654.46, 0.014]  # pb
total_events_bg = [5215.46, 6654.46, 1]
selection_scale_bg = [749.77/5215.46, 1591.09/6654.46, 1]
files_list_bg = [
    [file_Zcc, chosen_variable[0], "Z #rightarrow cc", cross_sections_bg[0], total_events_bg[0], selection_scale_bg[0]],
    [file_Zbb, chosen_variable[0], "Z #rightarrow bb", cross_sections_bg[1], total_events_bg[1], selection_scale_bg[1]],
    [file_4body, chosen_variable[0], "Z #rightarrow e #nu qq", cross_sections_bg[2], total_events_bg[2], selection_scale_bg[2]]
]

legend_list_bg = [f[2] for f in files_list_bg]
ratio_list_bg = [f[2] for f in files_list_bg]

legend_list_signal = [f[2] for f in files_list_signal]
ratio_list_signal = [f[2] for f in files_list_signal]

colors_signal = [ROOT.kMagenta - 7, ROOT.kMagenta - 2, ROOT.kMagenta + 3]
colors_bg = [856, 410, 801, 629, 879, 602, 921, 622]

def make_hist_from_ntuple(files_list):
    h_list = []
    for f in files_list:
        print("Looking at file", f[2])
        my_file = ROOT.TFile.Open(f[0], "READ")
        if not my_file:
            print("Failed to open file:", f[0])
            continue
        tree = my_file.Get("events")
        if not tree:
            print("Tree 'events' not found in file", f[0])
            my_file.Close()
            continue

        hist_name = f"hist_{f[2]}"
        hist = ROOT.TH1F(hist_name, hist_name, 50, 0, 100)  # Adjust the number of bins and range as needed
        tree.Draw(f"{f[1]}>>{hist_name}")
        
        if normalisation:
            print("Normalising....")
            cross_section = f[3]
            events_generated = f[4]
            selected_events = hist.Integral()
            scaling_factor = (cross_section * luminosity) / events_generated * (selected_events / events_generated)
            print(f"Scale factor for {f[2]} = {scaling_factor}, with selection efficiency = {selected_events/events_generated} and expected events total as {(cross_section * luminosity) * selected_events/events_generated}")
            hist.Scale(scaling_factor)

        hist.SetDirectory(0)
        h_list.append(hist)
        my_file.Close()
        print("Histogram added to h_list")
    return h_list

def make_significance(files_list, n_bins, x_min, x_max, h_list_bg, significance_direction):
    sig_list = []
    for h in files_list:
        sig_hist = ROOT.TH1F("Significance", "Significance", n_bins, x_min, x_max)

        if significance_direction == "RL":
            bin_range = range(1, n_bins + 1)
        elif significance_direction == "LR":
            bin_range = range(n_bins, 0, -1)
        else:
            raise ValueError("Invalid significance direction. Choose 'LR' or 'RL'.")

        cumulative_signal = 0
        cumulative_background = 0

        for bin_idx in bin_range:
            cumulative_signal += h.Integral(bin_idx, bin_idx)
            cumulative_background += sum(bg_hist.Integral(bin_idx, bin_idx) for bg_hist in h_list_bg)
            print(f"sig direction is {significance_direction} ; idx is {bin_idx} sig is {cumulative_signal}, bkg is {cumulative_background}")
            sigma = cumulative_background * uncertainty_count_factor
            significance = 0
            if cumulative_signal + cumulative_background > 0 and cumulative_background > 0 and cumulative_signal != 0 and sigma != 0:
                n = cumulative_signal + cumulative_background
                significance = math.sqrt(abs(
                    2 * (n * math.log((n * (cumulative_background + sigma**2)) / (cumulative_background**2 + n * sigma**2)) - 
                         (cumulative_background**2 / sigma**2) * math.log((1 + (sigma**2 * (n - cumulative_background)) / 
                                                                          (cumulative_background * (cumulative_background + sigma**2))))
                )))
            sig_hist.SetBinContent(bin_idx, significance)
        sig_list.append(sig_hist)
    return sig_list

h_list_signal = make_hist_from_ntuple(files_list_signal)
h_list_bg = make_hist_from_ntuple(files_list_bg)
n_bins = h_list_bg[0].GetNbinsX()
x_min = h_list_bg[0].GetXaxis().GetXmin()
x_max = h_list_bg[0].GetXaxis().GetXmax()
h_list_significance = make_significance(h_list_signal, n_bins, x_min, x_max, h_list_bg, significance_direction)

## Just some statistics:
n_signal_20 = h_list_signal[0].Integral()
n_signal_50 = h_list_signal[1].Integral()
n_signal_70 = h_list_signal[2].Integral()

print(h_list_bg[0].Integral(), 'Number of Z->cc events')
print(h_list_bg[1].Integral(), "Number of Z->bb events")
print(h_list_bg[2].Integral(), "Number of Z->4body events")
print(h_list_signal[0].Integral(), "Number of 20 GeV events")
print(h_list_signal[1].Integral(), "Number of 50 GeV events")
print(h_list_signal[2].Integral(), "Number of 70 GeV events")
n_background_bb = h_list_bg[1].Integral()

def make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, h_list_significance):
    c = ROOT.TCanvas("can","can",600,600)   
    pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.25, 1.0, 1.0, 21)  # Adjust the top and bottom margins
    pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.24, 22) #the significance panel
    
    pad1.SetFillColor(0)
    pad1.SetBottomMargin(0.02)
    
    if log_scale == True:
        pad1.SetLogy()
    pad1.SetTickx()
    pad1.SetTicky()
    pad1.Draw()

    pad2.SetFillColor(0)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.4)
    pad2.Draw()
    
    # Background Legend
    leg_bg = ROOT.TLegend(0.15, 0.6, 0.35, 0.75)
    leg_bg.SetFillStyle(0)
    leg_bg.SetLineWidth(0)

    # Signal Legend
    leg_sig = ROOT.TLegend(0.15, 0.4, 0.35, 0.55)
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
        h.SetLineWidth(3)
        h.SetStats(0)
        if not log_scale and not normalisation:
            h.GetYaxis().SetTitle("Entries")
        elif log_scale and not normalisation:
            h.GetYaxis().SetTitle("Log Entries")
        elif not log_scale and normalisation:
            h.GetYaxis().SetTitle("Normalised Entries")
        elif log_scale and normalisation:
            h.GetYaxis().SetTitle("Log Normalised Entries")

        h.GetXaxis().SetTitleSize(0.03)
        h.GetYaxis().SetTitleSize(0.03)
        h.GetXaxis().SetLabelOffset(-1000)
        h.GetYaxis().SetTitleOffset(1.4)
        h.SetMaximum(1.25 * h_max)
        h.SetMinimum(0.1)
        h.Draw('hist same')

    for ih,h in enumerate(h_list_bg):   
        h.SetLineColor(colors_bg[ih])
        h.SetLineWidth(3)
        h.SetStats(0)
        if not log_scale and not normalisation:
            h.GetYaxis().SetTitle("Entries")
        elif log_scale and not normalisation:
            h.GetYaxis().SetTitle("Log Entries")
        elif not log_scale and normalisation:
            h.GetYaxis().SetTitle("Normalised Entries")
        elif log_scale and normalisation:
            h.GetYaxis().SetTitle("Log Normalised Entries")

        h.GetXaxis().SetTitleSize(0.03)
        h.GetYaxis().SetTitleSize(0.03)
        h.GetXaxis().SetLabelOffset(-1000)
        h.GetYaxis().SetTitleOffset(1.4)
        h.SetMaximum(1.25 * h_max)
        h.SetMinimum(0.1)
        h.Draw('hist same')
    
    pad1.RedrawAxis()
    leg_sig.Draw()
    leg_bg.Draw()

    text_title = ROOT.TLatex()
    text_title.SetTextSize(0.04)
    text_title.SetTextFont(42)
    text_title.DrawLatexNDC(0.10, 0.92, "#font[72]{FCCee} Simulation (DELPHES)")

    text_selection = ROOT.TLatex()
    text_selection.SetTextSize(0.03)
    text_selection.SetTextFont(42)
    text_selection.DrawLatexNDC(0.60,  0.8, "#font[52]{No Selection}")

    text_lumi = ROOT.TLatex()
    text_lumi.SetTextSize(0.03)
    text_lumi.SetTextFont(42)
    text_lumi.DrawLatexNDC(0.60, 0.75, "#font[52]{#sqrt{s} = 91 GeV , #int L dt = 10 fb^{-1}}")

    pad1.RedrawAxis()

    pad2.cd()
    for i,h in enumerate(h_list_significance):
        
        h.SetLineColor(colors_signal[i])
        h.SetLineWidth(2)
        h.SetStats(0)
        
        h.GetYaxis().SetTitle("Z")
        h.GetYaxis().SetTitleSize(0.8)
        h.GetYaxis().SetTitleOffset(1.4)
        h.GetYaxis().SetLabelSize(h.GetYaxis().GetLabelSize() * 2)
        h.GetYaxis().SetLabelOffset(0.02)
        
        h.GetXaxis().SetTitle(f"{chosen_variable[1]}")
        h.GetXaxis().SetTitleSize(0.12)
        h.GetXaxis().SetTitleOffset(1.2)
        h.GetXaxis().SetLabelSize(0.1)
        h.GetXaxis().SetLabelOffset(0.02)
        h.GetXaxis().SetRangeUser(0, 50)
        if i==0:
            h.Draw()
        else:
            h.Draw('hist same')

    if log_scale and normalisation:
        c.SetLogy(log_scale)
        c.SaveAs(output_dir + "chi2" + selection + chosen_variable[0] + "log_" + "norm" + ".pdf", "R")
    elif log_scale and not normalisation:
        c.SetLogy(log_scale)
        c.SaveAs(output_dir + "BackgroundVSignal_" + selection + chosen_variable[0] + "log" + ".pdf", "R")
    elif normalisation and not log_scale:        
        c.SaveAs(output_dir + "BackgroundVSignal_" + selection + chosen_variable[0] + "norm" + ".pdf", "R")
    else:
        c.SaveAs(output_dir + "BackgroundVSignal_" + selection + chosen_variable[0] + ".pdf", "R")
    
    return

make_plot(h_list_signal, h_list_bg, legend_list_signal, legend_list_bg, h_list_significance)
