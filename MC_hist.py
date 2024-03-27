import ROOT

def create_histogram(file_paths, tree_name, variable_name, hist_params, label):
    """
    Create a histogram for a given variable from multiple ROOT files.

    Parameters:
    - file_paths: list of strings, paths to ROOT files
    - tree_name: string, name of the TTree
    - variable_name: string, name of the variable to plot
    - hist_params: tuple, parameters for the histogram (name, title, bins, x_min, x_max)
    - label: string, label for the histogram

    Returns:
    - hist: ROOT.TH1F object, the filled histogram
    """
    hist = ROOT.TH1F(hist_params[0], hist_params[1], hist_params[2], hist_params[3], hist_params[4])
    for file_path in file_paths:
        f = ROOT.TFile.Open(file_path)
        tree = f.Get(tree_name)
        for event in tree:
            value = getattr(event, variable_name)
            hist.Fill(value)
        f.Close()
    hist.SetTitle(label)
    return hist

# Define your files and parameters here
background_files = ["/eos/user/t/tcritchl/MCfilter/p8_ee_Zbb_ecm91/chunk_0.root", "/eos/user/t/tcritchl/MCfilter/p8_ee_Zcc_ecm91/chunk_1.root", "/eos/user/t/tcritchl/MCfilter/ejjnu/chunk_0.root"]
signal_files = ["/eos/user/t/tcritchl/MCfilter/HNL_Dirac_ejj_20GeV_1e-3Ve/chunk_0.root", "/eos/user/t/tcritchl/MCfilter/HNL_Dirac_ejj_50GeV_1e-3Ve/chunk_0.root", "/eos/user/t/tcritchl/MCfilter/HNL_Dirac_ejj_70GeV_1e-3Ve/chunk_0.root"]
tree_name = "events"
variable_name = "RecoElectron_lead_eta"
hist_params = ("hist", "Histogram of RecoElectron_lead_eta;eta;Events", 50, -2.5, 2.5)

# Create histograms
hist_background = create_histogram(background_files, tree_name, variable_name, hist_params, "Background")
hist_signal = create_histogram(signal_files, tree_name, variable_name, hist_params, "Signal")

# Plotting
c = ROOT.TCanvas("c", "canvas", 800, 600)
hist_background.SetLineColor(ROOT.kBlue)
hist_signal.SetLineColor(ROOT.kRed)
hist_background.Draw()
hist_signal.Draw("SAME")
c.BuildLegend()
c.SaveAs("comparison_plot.pdf")
