import ROOT

def create_histogram(file_path, tree_name, variable_names, hist_params, label, color):

    hist1 = ROOT.TH1F(hist_params[0] + "_" + label + "_1", hist_params[1], hist_params[2], hist_params[3], hist_params[4])
    hist2 = ROOT.TH1F(hist_params[0] + "_" + label + "_2", hist_params[1], hist_params[2], hist_params[3], hist_params[4])
    f = ROOT.TFile.Open(file_path)
    tree = f.Get(tree_name)
    for event in tree:
        # Assuming both variables are RVec<float>
        value1_attr = getattr(event, variable_names[0])
        value1 = value1_attr[0] if value1_attr.size() > 0 else float('nan')  # Use NaN for empty RVec
        
        value2_attr = getattr(event, variable_names[1])
        value2 = value2_attr[0] if value2_attr.size() > 0 else float('nan')  # Use NaN for empty RVec
        
        # Fill the histograms
        if not ROOT.TMath.IsNaN(value1): hist1.Fill(value1)
        if not ROOT.TMath.IsNaN(value2): hist2.Fill(value2)

    f.Close()

    for hist in (hist1, hist2):
        hist.SetLineColor(color)
    hist2.SetLineStyle(7)  # Set dashed line for the second variable
    return hist1, hist2

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
variable_names = ("FSGenElectron_theta", "RecoElectron_theta")
hist_params = ("theta", "Theta distribution;Theta (rad);Events", 50, -ROOT.TMath.Pi(), ROOT.TMath.Pi())
# Create histograms for each file and variable
histograms = []
for file_path, label, color in background_files + signal_files:
    hist1, hist2 = create_histogram(file_path, tree_name, variable_names, hist_params, label, color, linestyle=2 if "Reco" in label else 1)
    histograms.extend([hist1, hist2])

# Plotting setup
c = ROOT.TCanvas("c", "canvas", 1000, 800)  # Bigger canvas for better readability
legend = ROOT.TLegend(0.1, 0.7, 0.3, 0.9)  # Adjusted position for visibility
legend.SetTextSize(0.03)  # Bigger text size

max_y = max([hist.GetMaximum() for hist in histograms]) * 1.2

for hist in histograms:
    draw_option = "HIST SAME" if hist != histograms[0] else "HIST"
    hist.SetMaximum(max_y)
    hist.Draw(draw_option)
    legend_entry_label = hist.GetTitle() + (" (dashed)" if "Reco" in hist.GetTitle() else " (solid)")
    legend.AddEntry(hist, legend_entry_label, "l")

legend.Draw()
c.Modified()
c.Update()
c.SaveAs("comparison_plot_theta_distribution.pdf")
