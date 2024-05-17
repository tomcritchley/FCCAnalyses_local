import ROOT

def create_histogram(file_path, tree_name, variable_names, hist_params, label, color, color2):
    # Creating histograms with more descriptive names
    hist1 = ROOT.TH1F(f"{hist_params[0]}_{label}_truth","", hist_params[2], hist_params[3], hist_params[4])
    hist2 = ROOT.TH1F(f"{hist_params[0]}_{label}_reco","", hist_params[2], hist_params[3], hist_params[4])

    f = ROOT.TFile.Open(file_path)
    tree = f.Get(tree_name)
    for event in tree:
        if event.n_FSGenElectron > 0:  # Ensuring at least one generated electron
            for i in range(event.n_FSGenElectron):
                eta = abs(event.FSGenElectron_eta[i])
                energy = event.FSGenElectron_e[i]
                pt = event.FSGenElectron_pt[i]
                # Apply the DELPHES efficiency conditions
                if energy > 2.0 and pt >= 0.1 and eta <= 2.56: #--> the DELPHES condition
                    value1_attr = getattr(event, variable_names[0], None)
                    value1 = value1_attr[0] if value1_attr.size() > 0 else float('nan')
                    value2_attr = getattr(event, variable_names[1], None)
                    value2 = value2_attr[0] if value2_attr.size() > 0 else float('nan')
                    
                    if not ROOT.TMath.IsNaN(value1): hist1.Fill(value1)
                    if not ROOT.TMath.IsNaN(value2): hist2.Fill(value2)
    f.Close()

    hist1.SetLineColor(color)
    hist1.SetStats(0)
    hist1.SetLineWidth(2)
    hist2.SetLineColor(color2)
    hist2.SetLineStyle(1)  # Solid line for reconstructed data
    hist2.SetStats(0)
    hist2.SetLineWidth(2)
    return hist1, hist2

# File and parameters for Zbb
file_path = "/eos/user/t/tcritchl/MCfilter/p8_ee_Zbb_ecm91/chunk_0.root" #p8_ee_Zbb_ecm91
label = "z-->bb"
color = ROOT.kBlue  # Color for truth data
color2 = ROOT.kRed  # Color for reco data

tree_name = "events"
variable_names = ("FSGenElectron_e", "RecoElectron_e")
hist_params = ("pt", "Energy distribution;Eta;Events", 100, 0, 50)  # Updated axis labels and range
#hist_params = ("pt", "pt distribution;pt;Events", 100, -ROOT.TMath.Pi(), ROOT.TMath.Pi())

# Create histograms for Zbb truth and reco
hist1, hist2 = create_histogram(file_path, tree_name, variable_names, hist_params, label, color, color2)

# Create canvas and draw histograms
c = ROOT.TCanvas("c", "canvas", 1400, 1000)

hist1.Draw("HIST")
hist2.Draw("HISTSAME")

legend = ROOT.TLegend(0.6, 0.75, 0.8, 0.85)
legend.SetFillStyle(0)
legend.SetLineWidth(0)
legend.SetTextSize(0.03) 
legend.AddEntry(hist1, r"Truth Z #rightarrow bb", "l")
legend.AddEntry(hist2, r"Reconstructed Z #rightarrow bb", "l")
legend.Draw()

hist1.GetXaxis().SetTitle(r"Electron  Energy [GeV]")
hist1.GetYaxis().SetTitle("Entries")
hist1.GetXaxis().SetTitleSize(0.04)
hist1.GetYaxis().SetTitleSize(0.04)
hist1.GetXaxis().SetLabelSize(0.03)
hist1.GetYaxis().SetLabelSize(0.03)

# Adding text labels
text_title = ROOT.TLatex()
text_title.SetTextSize(0.04)
text_title.SetTextFont(42)
text_title.DrawLatexNDC(0.1, 0.92, "#font[72]{FCCee} Simulation (DELPHES)")

text_selection = ROOT.TLatex()
text_selection.SetTextSize(0.025)
text_selection.SetTextFont(42)
text_selection.DrawLatexNDC(0.6,  0.92,  r"E_{electron} #geq 2.0 GeV, p_{T} #geq 0.1 GeV, | #eta| #leq 2.56")
#text_selection.DrawLatexNDC(0.61,  0.72,  "#font[52]{No Selection}")


c.SaveAs(f"/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/generator_plots/Zbb_energy_Selection.pdf")
