import ROOT

def create_histogram(file_info, tree_name, variable_names, hist_params):
    histograms = []

    for file_path, label, color in file_info:
        hist1 = ROOT.TH1F(hist_params[0] + "_" + label + "_1", hist_params[1], hist_params[2], hist_params[3], hist_params[4])
        hist2 = ROOT.TH1F(hist_params[0] + "_" + label + "_2", hist_params[1], hist_params[2], hist_params[3], hist_params[4])

        hist1.SetLineColor(color)
        hist1.SetLineStyle(1)
        hist2.SetLineColor(color)
        hist2.SetLineStyle(2)

        f = ROOT.TFile.Open(file_path)
        tree = f.Get(tree_name)
        for event in tree:
            value1_attr = getattr(event, variable_names[0])
            value2_attr = getattr(event, variable_names[1])

            # Using a general approach to check for iterable attributes
            value1 = value1_attr[0] if hasattr(value1_attr, '__iter__') and len(value1_attr) > 0 else value1_attr
            value2 = value2_attr[0] if hasattr(value2_attr, '__iter__') and len(value2_attr) > 0 else value2_attr

            hist1.Fill(value1)
            hist2.Fill(value2)
        f.Close()

        histograms.append((hist1, hist2))

    return histograms

# Your file paths, labels, and colors
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

# Create and plot histograms
histograms_background = create_histogram(background_files, tree_name, variable_names, hist_params)
histograms_signal = create_histogram(signal_files, tree_name, variable_names, hist_params)

c = ROOT.TCanvas("c", "Canvas", 800, 600)
legend = ROOT.TLegend(0.1, 0.7, 0.3, 0.9)

# Adjusted plotting code for histograms

c.SaveAs("comparison_plot_with_multiple_variables.pdf")
