import ROOT

def create_histogram(file_path, tree_name, variable_names, hist_params, label, color, color2):
    # Creating histograms with more descriptive names
    hist1 = ROOT.TH1F(f"{hist_params[0]}_{label}_truth", f"{label}", hist_params[2], hist_params[3], hist_params[4])
    hist2 = ROOT.TH1F(f"{hist_params[0]}_{label}_reco", f"Reco: {label}", hist_params[2], hist_params[3], hist_params[4])

    f = ROOT.TFile.Open(file_path)
    tree = f.Get(tree_name)
    for event in tree:
        if event.n_FSGenElectron > 0:  # Ensuring at least one generated electron
            value1_attr = getattr(event, variable_names[0], None)
            value1 = value1_attr[0] if value1_attr.size() > 0 else float('nan')
            value2_attr = getattr(event, variable_names[1], None)
            value2 = value2_attr[0] if value2_attr.size() > 0 else float('nan')
            
            if not ROOT.TMath.IsNaN(value1): hist1.Fill(value1)
            if not ROOT.TMath.IsNaN(value2): hist2.Fill(value2)

    f.Close()

    hist1.SetLineColor(color)
    hist1.SetStats(0)
    hist2.SetLineColor(color2)
    hist2.SetLineStyle(1)  # Solid line for reconstructed data
    hist2.SetStats(0)
    return hist1, hist2

# File and parameters for Zbb
file_path = "/eos/user/t/tcritchl/MCfilter/p8_ee_Zbb_ecm91/chunk_0.root" #p8_ee_Zbb_ecm91
label = "z-->bb"
color = ROOT.kBlue  # Color for truth data
color2 = ROOT.kRed  # Color for reco data

tree_name = "events"
variable_names = ("FSGenElectron_pt", "RecoElectron_pt")
hist_params = ("pt", "pt distribution;pt;Events", 100, 2, 50)

# Create histograms for Zbb truth and reco
hist1, hist2 = create_histogram(file_path, tree_name, variable_names, hist_params, label, color, color2)

# Create canvas and draw histograms
c = ROOT.TCanvas("c", "canvas", 1200, 800)
hist1.Draw("HIST")
hist2.Draw("HISTSAME")

# Adding a legend
legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry(hist1, "Truth: Z->bb", "l")
legend.AddEntry(hist2, "Reco: Z->bb", "l")
legend.Draw()

# Adding text labels
text_title = ROOT.TLatex()
text_title.SetTextSize(0.04)
text_title.SetTextFont(42)
text_title.DrawLatexNDC(0.1, 0.92, "#font[72]{FCCee} Simulation (DELPHES)")

c.SaveAs(f"/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/generator_plots/bb_electron_pt.pdf")
