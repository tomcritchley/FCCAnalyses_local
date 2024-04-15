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

        if not ROOT.TMath.IsNaN(value1) and value1 != 4:
            hist1.Fill(value1)
        if not ROOT.TMath.IsNaN(value2) and value2 != 4:
            hist2.Fill(value2)

    f.Close()

    hist1.SetLineColor(color)
    hist1.SetStats(0)
    hist2.SetLineColor(color2)
    hist2.SetLineStyle(0)  # Dashed line for reconstructed data
    hist2.SetStats(0)
    return hist1, hist2

# File and parameters for Zbb
file_path = "/eos/user/t/tcritchl/MCfilter/p8_ee_Zbb_ecm91/chunk_0.root"
label = "z->cc"
color = ROOT.kBlue
color2 = ROOT.kRed

tree_name = "events"
variable_names = ("FSGenElectron_e", "RecoElectron_e")
hist_params = ("pt", "pt distribution;pt;Events", 100, 0, 50)

# Create histograms for Zbb truth and reco
hist1, hist2 = create_histogram(file_path, tree_name, variable_names, hist_params, label, color)

# Create canvas and draw histograms
c = ROOT.TCanvas("c", "canvas", 1200, 800)
hist1.Draw("HIST")
hist2.Draw("HISTSAME")

# Adding a legend
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry(hist1, "Truth: Z->cc", "l")
legend.AddEntry(hist2, "Reco: Z->cc", "l")
legend.Draw()

# Adding text labels
text_title = ROOT.TLatex()
text_title.SetTextSize(0.04)
text_title.SetTextFont(42)
text_title.DrawLatexNDC(0.1, 0.92, "#font[72]{FCCee} Simulation (DELPHES)")

c.SaveAs("bb_electron_e_comparison.pdf")
