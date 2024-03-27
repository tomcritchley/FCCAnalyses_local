import ROOT

def create_histogram(file_info, tree_name, variable_names, hist_params):
    """
    Create and fill histograms for given variables from multiple ROOT files.
    
    Parameters:
    - file_info: List of tuples, each containing (file_path, label, color)
    - tree_name: string, name of the TTree
    - variable_names: tuple of strings, names of the variables to plot
    - hist_params: tuple, parameters for the histogram (name, title, bins, x_min, x_max)
    """
    histograms = []

    for file_path, label, color in file_info:
        # Initialize histograms for each file
        hist1 = ROOT.TH1F(hist_params[0] + "_" + label + "_1", hist_params[1], hist_params[2], hist_params[3], hist_params[4])
        hist2 = ROOT.TH1F(hist_params[0] + "_" + label + "_2", hist_params[1], hist_params[2], hist_params[3], hist_params[4])

        # Set histogram styles
        hist1.SetLineColor(color)
        hist1.SetLineStyle(1)  # Solid line for the first variable
        hist2.SetLineColor(color)
        hist2.SetLineStyle(2)  # Dashed line for the second variable

        f = ROOT.TFile.Open(file_path)
        tree = f.Get(tree_name)
        for event in tree:
            value1 = getattr(event, variable_names[0])[0] if getattr(event, variable_names[0]).size() > 0 else 0
            value2 = getattr(event, variable_names[1])[0] if getattr(event, variable_names[1]).size() > 0 else 0
            hist1.Fill(value1)
            hist2.Fill(value2)
        f.Close()

        histograms.append((hist1, hist2))

    return histograms

# Define your files and parameters here
background_files = [
    ("/eos/user/t/tcritchl/MCfilter/p8_ee_Zbb_ecm91/chunk_0.root", "Zbb", ROOT.kBlue),
    ("/eos/user/t/tcritchl/MCfilter/p8_ee_Zcc_ecm91/chunk_1.root", "Zcc", ROOT.kGreen),
    ("/eos/user/t/tcritchl/MCfilter/ejjnu/chunk_0.root", "ejjnu", ROOT.kCyan)
]
signal_files = [
    ("/eos/user/t/tcritchl/MCfilter/HNL_Dirac_ejj_20GeV_1e-3Ve/chunk_0.root", "HNL_20GeV", ROOT.kRed),
    ("/eos/user/t/tcritchl/MCfilter/HNL_Dirac_ejj_50GeV_1e-3Ve/chunk_0.root", "HNL_50GeV", ROOT.kMagenta),
    ("/eos/user/t/tcritchl/MCfilter/HNL_Dirac_ejj_70GeV_1e-3Ve/chunk_0.root", "HNL_70GeV", ROOT.kOrange)
]
tree_name = "events"
variable_names = ("RecoElectron_lead_eta", "FSGenElectron_eta")
hist_params = ("hist", "Histogram of RecoElectron_lead_eta and FSGenElectron_eta;eta;Events", 50, -2.5, 2.5)

# Create histograms
histograms_background = create_histogram(background_files, tree_name, variable_names, hist_params)
histograms_signal = create_histogram(signal_files, tree_name, variable_names, hist_params)

# Plotting
c = ROOT.TCanvas("c", "Canvas", 800, 600)
legend = ROOT.TLegend(0.1, 0.7, 0.3, 0.9)

for hists in histograms_background + histograms_signal:
    hists[0].Draw("HIST SAME")
    hists[1].Draw("HIST SAME")
    legend.AddEntry(hists[0], hists[0].GetTitle(), "l")
    legend.AddEntry(hists[1], hists[1].GetTitle(), "l")

legend.Draw()
c.SaveAs("comparison_plot_with_multiple_variables.pdf")
