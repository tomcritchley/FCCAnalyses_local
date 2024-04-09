import ROOT

def create_histogram(file_path, tree_name, variable_names, hist_params, label, color):
    # Creating histograms with more descriptive names
    hist1 = ROOT.TH1F(f"{hist_params[0]}_{label}_truth", f"Truth: {label}", hist_params[2], hist_params[3], hist_params[4])
    hist2 = ROOT.TH1F(f"{hist_params[0]}_{label}_reco", f"Reco: {label}", hist_params[2], hist_params[3], hist_params[4])

    f = ROOT.TFile.Open(file_path)
    tree = f.Get(tree_name)
    for event in tree:
        value1_attr = getattr(event, variable_names[0], None)
        value1 = value1_attr[0] if value1_attr.size() > 0 else float('nan')
        value2_attr = getattr(event, variable_names[1], None)
        value2 = value2_attr[0] if value2_attr.size() > 0 else float('nan')
        
        if not ROOT.TMath.IsNaN(value1): hist1.Fill(value1)
        if not ROOT.TMath.IsNaN(value2): hist2.Fill(value2)

    f.Close()

    hist1.SetLineColor(color)
    hist1.SetStats(0)
    hist2.SetLineColor(color)
    hist2.SetLineStyle(7)  # Dashed line for reconstructed data
    hist2.SetStats(0)
    return hist1, hist2

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
variable_names = ("FSGenElectron_e", "RecoElectron_e")
hist_params = ("theta", "Theta distribution;Theta (rad);Events", 50, 0, 50)

histograms = []
for file_path, label, color in background_files + signal_files:
    hist1, hist2 = create_histogram(file_path, tree_name, variable_names, hist_params, label, color)
    histograms.append((hist1, "Truth: " + label))
    histograms.append((hist2, "Reco: " + label))

max_y = max([hist[0].GetMaximum() for hist in histograms]) * 1.2

c = ROOT.TCanvas("c", "canvas", 1200, 800)  # Adjusted canvas size for better visibility
c.SetLogy(1)
# Adjusted legend size and position for better readability
legend = ROOT.TLegend(0.1, 0.5, 0.4, 0.9)  # Enlarged and repositioned legend
legend.SetTextSize(0.02)  # Reduced text size for more entries


for hist, name in histograms:
    hist.SetMaximum(max_y)
    draw_option = "HIST SAME"
    hist.Draw(draw_option)
    legend.AddEntry(hist, name, "l")

legend.Draw()

text_title = ROOT.TLatex()
text_title.SetTextSize(0.04)
text_title.SetTextFont(42)
text_title.DrawLatexNDC(0.70, 0.82, "#font[72]{FCCee} Simulation (DELPHES)")

text_selection = ROOT.TLatex()
text_selection.SetTextSize(0.03)
text_selection.SetTextFont(42)
text_selection.DrawLatexNDC(0.70, 0.77, "#font[52]{No Selection}")

text_lumi = ROOT.TLatex()
text_lumi.SetTextSize(0.03)
text_lumi.SetTextFont(42)
text_lumi.DrawLatexNDC(0.70, 0.72, "#font[52]{#sqrt{s} = 91 GeV , #int L dt = 10 fb^{-1}}")

c.SaveAs("comparison_plot_variables_energy.pdf")
