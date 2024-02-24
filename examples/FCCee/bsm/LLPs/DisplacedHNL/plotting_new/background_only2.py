import ROOT
from ROOT import *
import os

selection = "selNone"
variable = "RecoLeadJet_e"
normalisation = True  # Set this to True to enable normalization
luminosity = 10000  # 10 fb^-1 integrated luminosity
log_scale = True
input_dir = "/eos/user/t/tcritchl/outputs/output_final/testingAll/"
output_dir = "Background/" + selection + '/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Directory ", output_dir, " Created ")
else:
    print("Directory ", output_dir, " already exists")

# Define the files
file_Zud = input_dir + 'p8_ee_Zud_ecm91_' + selection + '_histo.root'
file_Zcc = input_dir + 'p8_ee_Zcc_ecm91_' + selection + '_histo.root'
file_Zbb = input_dir + 'p8_ee_Zbb_ecm91_' + selection + '_histo.root'

# Define the unique cross sections and total events for each process
cross_sections = [11870.5, 5215.46, 6654.46]  # Cross sections in pb
total_events = [497.658684, 499.786495, 438.738637]  # Total events generated for each process --> but multiplied by the fraction used

# Input list of background signals
files_list_bg = [
    [file_Zud, variable, "Z -> ud", cross_sections[0], total_events[0]],
    [file_Zcc, variable, "Z -> cc", cross_sections[1], total_events[1]],
    [file_Zbb, variable, "Z -> bb", cross_sections[2], total_events[2]]
]

# Add the third element of the list above and input them as the legend

legend_list_bg = [f[2] for f in files_list_bg]

colors_bg = [856, 410, 801, 629, 879, 602, 921, 622]

def make_hist(files_list):
    
    h_list = []
    for f in files_list:
        print("Looking at file", f[2])
        my_file = ROOT.TFile.Open(f[0])  # Open the root file
        print("Getting histogram for variable", f[1])
        hist = my_file.Get(f[1])  # Select the chosen variable from the histo root file

        if normalisation:
            # Apply normalization based on cross section, total events, and luminosity
            cross_section = f[3]  # Cross section in pb
            events_generated = f[4]  # Total events generated
            scaling_factor = (cross_section * luminosity) / events_generated
            hist.Scale(scaling_factor)

        hist.SetDirectory(0)  # Make the chosen histogram independent of the directory
        h_list.append(hist)
        print("Histogram added to h_list")
        my_file.Close()
        print("-----------------------")
    return h_list

h_list_bg = make_hist(files_list_bg)

def make_plot(h_list_bg, legend_list_bg):
    
    c = ROOT.TCanvas("can", "can", 800, 600)  # Create a square canvas
    
    leg_bg = ROOT.TLegend(0.13, 0.75, 0.30, 0.89)    
    leg_bg.SetFillStyle(0)
    leg_bg.SetLineWidth(0)

    h_max = 0
    
    for ih, h in enumerate(h_list_bg):
        if h.GetMaximum() > h_max:
            h_max = h.GetMaximum()
        h.Sumw2()
    for ih, h in enumerate(h_list_bg):
        leg_bg.AddEntry(h, legend_list_bg[ih])

    # Draw in the canvas
    c.cd()
    for ih, h in enumerate(h_list_bg):
        h.SetLineColor(colors_bg[ih])
        h.SetLineWidth(3)
        h.SetStats(0)
        if not log_scale and not normalisation:
            #h.SetTitle("FCCee Simulation")
            h.GetXaxis().SetTitle(f"{variable}")
            h.GetYaxis().SetTitle("Entries")
        elif log_scale and not normalisation:
            #h.SetTitle("FCCee Simulation")
            h.GetXaxis().SetTitle(f"{variable}")
            h.GetYaxis().SetTitle("Log Entries")
        elif not log_scale and normalisation:
            #h.SetTitle("FCCee Simulation")
            h.GetXaxis().SetTitle(f"{variable}")
            h.GetYaxis().SetTitle("Normalised Entries")
        elif log_scale and normalisation:
            #h.SetTitle("FCCee Simulation")
            h.GetXaxis().SetTitle(f"{variable}")
            h.GetYaxis().SetTitle("Log Normalised Entries")

        h.GetXaxis().SetTitleSize(0.03)  # Adjust the text size for X-axis label
        h.GetYaxis().SetTitleSize(0.03)  # Adjust the text size for Y-axis label
        h.GetXaxis().SetTitleOffset(1.4)  # Adjust the position of the X-axis label
        h.GetYaxis().SetTitleOffset(1.4)  # Adjust the position of the Y-axis label
        h.SetMaximum(1.25 * h_max)
        h.Draw('hist same')
        
    leg_bg.Draw()

    text_title = ROOT.TLatex()
    text_title.SetTextSize(0.04)
    text_title.SetTextFont(42)
    text_title.DrawLatexNDC(0.52, 0.85, "#font[72]{FCCee} Simulation (DELPHES)")

    text_selection = ROOT.TLatex()
    text_selection.SetTextSize(0.03)
    text_selection.SetTextFont(42)
    text_selection.DrawLatexNDC(0.52, 0.80, "#font[52]{No Selection}")

    text_lumi = ROOT.TLatex()
    text_lumi.SetTextSize(0.03)
    text_lumi.SetTextFont(42)
    text_lumi.DrawLatexNDC(0.52, 0.75, "#font[52]{#sqrt{s} = 91 GeV , #int L dt = 10 fb^{-1}}")
    """
    text_e_cm = ROOT.TLatex()
    text_e_cm.SetTextSize(0.03)
    text_e_cm.SetTextFont(42)
    text_e_cm.DrawLatexNDC(0.16, 0.70, "#font[52]{#sqrt{s} = 91 GeV}")
    """    
    if log_scale and normalisation:
        c.SetLogy(log_scale)
        c.SaveAs(output_dir + "Background_" + selection + variable + "log_" + "norm" + ".pdf", "R")
    elif log_scale and not normalisation:
        c.SetLogy(log_scale)
        c.SaveAs(output_dir + "Background_" + selection + variable + "log" + ".pdf", "R")
    elif normalisation and not log_scale:        
        c.SaveAs(output_dir + "Background_" + selection + variable + "norm" + ".pdf", "R")
    else:
        c.SaveAs(output_dir + "Background_" + selection + variable + ".pdf", "R")
    
    return

make_plot(h_list_bg, legend_list_bg)