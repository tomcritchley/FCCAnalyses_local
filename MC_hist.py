import ROOT

def create_histogram(file_path, tree_name, variable_name, hist_params, label, color):
    """
    Create a histogram for a given variable from a ROOT file.

    Parameters:
    - file_path: string, path to the ROOT file
    - tree_name: string, name of the TTree
    - variable_name: string, name of the variable to plot
    - hist_params: tuple, parameters for the histogram (name, title, bins, x_min, x_max)
    - label: string, label for the histogram
    - color: ROOT color constant, color of the histogram

    Returns:
    - hist: ROOT.TH1F object, the filled histogram
    """
    hist = ROOT.TH1F(hist_params[0] + "_" + label, hist_params[1], hist_params[2], hist_params[3], hist_params[4])
    f = ROOT.TFile.Open(file_path)
    tree = f.Get(tree_name)
    for event in tree:
        value = getattr(event, variable_name)
        hist.Fill(value)
    f.Close()
    hist.SetLineColor(color)
    hist.SetTitle(label)
    return hist

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
variable_name = "RecoElectron_lead_eta"
hist_params = ("hist", "Histogram of RecoElectron_lead_eta;eta;Events", 50, -2.5, 2.5)

# Create histograms for each file
histograms = []
for file_path, label, color in background_files + signal_files:
    hist = create_histogram(file_path, tree_name, variable_name, hist_params, label, color)
    histograms.append(hist)

# Find the maximum y value among all histograms to adjust the y-axis range
max_y = max([hist.GetMaximum() for hist in histograms]) * 1.2  # Increase by 20% for some headroom

# Plotting
c = ROOT.TCanvas("c", "canvas", 800, 600)
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)

first_hist = True
for hist in histograms:
    hist.SetMaximum(max_y)  # Set the same y-axis range for all histograms
    draw_option = "HIST SAME" if not first_hist else "HIST"
    hist.Draw(draw_option)
    legend.AddEntry(hist, hist.GetTitle(), "l")
    first_hist = False

legend.Draw()
c.SaveAs("comparison_plot.pdf")
