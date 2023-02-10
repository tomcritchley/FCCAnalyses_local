import ROOT
import os
# plots don't appear while you run
ROOT.gROOT.SetBatch(0)
# no stat box on plots
ROOT.gStyle.SetOptStat(0)
# no title shown 
ROOT.gStyle.SetOptTitle(0)

HNL_mass = "50GeV"

#selection = "selGenEleEGt30"
#selection = "selJetPtGt20"
selection = "selNone"

output_dir =  HNL_mass + "_ejj_50k/"
output_dir_sel = HNL_mass + "_ejj_50k/" + selection +'/'
input_dir = "selected_hist/"

#input_file_Dirac = input_dir + 'histDirac_ejj_'+HNL_mass+'_AE10_'+selection+'.root'
#input_file_Majorana = input_dir + 'histDirac_ejj_'+HNL_mass+'_AE100_'+selection+'.root'

input_file_Dirac = input_dir + 'histDirac_ejj_'+HNL_mass+'_1e-3Ve_'+selection+'.root'
input_file_Majorana = input_dir + 'histMajorana_ejj_'+HNL_mass+'_1e-3Ve_'+selection+'.root'

# Set plot log-scale plots, default: False
log_scale = False

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:    
    print("Directory ",output_dir," already exists")

if not os.path.exists(output_dir_sel):
    os.mkdir(output_dir_sel)
    print("Directory ",output_dir_sel," Created ")
else:
    print("Directory ",output_dir_sel," already exists")
# list of lists
# each internal list: hist name, x title, y title, rebin (if needed)
variables_list = [
     ["FSGenElectron_pt", "FSGen electron pt", "Entries", 3],
     ["FSGenElectron_phi", "FSGen Electron phi", "Entries", 3],
     ["FSGenElectron_eta", "FSGen Electron eta", "Entries", 3],
     ["FSGenElectron_e", "FSGen Electron energy", "Entries", 3],

     ["RecoElectron_pt", "Reco electron pt", "Entries", 3],
     ["RecoElectron_phi", "Reco Electron phi", "Entries", 3],
     ["RecoElectron_theta", "Reco Electron theta", "Entries", 3],
     ["RecoElectron_e", "Reco Electron energy", "Entries", 3],
     ["RecoMissingEnergy_e", "Reco Missing_e", "Entries", 3],
     ["RecoMissingEnergy_pt", "Reco Missing_pt", "Entries", 3],
     ["RecoMissingEnergy_p", "Reco Missing_p", "Entries", 3],
     ["RecoMissingEnergy_px", "Reco Missing_px", "Entries", 3],
     ["RecoMissingEnergy_py", "Reco Missing_py", "Entries", 3],
     ["RecoMissingEnergy_pz", "Reco Missing_pz", "Entries", 3],
     ["RecoMissingEnergy_eta", "Reco Missing_eta", "Entries", 3],
     ["RecoMissingEnergy_theta", "Reco Missing_theta", "Entries", 3],
     ["RecoMissingEnergy_phi", "Reco Missing_phi", "Entries", 3],

     ["RecoHNL_electron_e", "Reco electron (-) energy", "Entries", 3],
     ["RecoHNL_electron_pt", "Reco electron (-) pt", "Entries", 3],
     ["RecoHNL_electron_phi", "Reco electron (-) phi", "Entries", 3],
     ["RecoHNL_electron_theta", "Reco electron (-) theta", "Entries", 3],
     ["RecoHNL_electron_eta", "Reco electron (-) energy", "Entries", 3],

     ["RecoHNL_positron_e", "Reco positron (+) energy", "Entries", 3],
     ["RecoHNL_positron_pt", "Reco positron (+) pt", "Entries", 3],
     ["RecoHNL_positron_phi", "Reco positron (+) phi", "Entries", 3],
     ["RecoHNL_positron_theta", "Reco positron (+) theta", "Entries", 3],
     ["RecoHNL_positron_eta", "Reco positron (+) energy", "Entries", 3],

     
     ["n_RecoJets", "Number of RecoJets", "Entries"],
     ["RecoJet_e", "Reco Jet energy", "Entries", 3],
     ["RecoJet_p", "Reco Jet p", "Entries", 3],
     ["RecoJet_pt", "Reco Jet pt", "Entries", 3],
     ["RecoJet_pz", "Reco Jet pz", "Entries", 3],
     ["RecoJet_eta", "Reco Jet eta", "Entries", 3],
     ["RecoJet_theta", "Reco Jet theta", "Entries", 3],
     ["RecoJet_phi", "Reco Jet phi", "Entries", 3],
     ["RecoJet_charge", "Reco Jet charge", "Entries", 3],
     ["RecoJetTrack_absD0", "Reco Jet abs_DO", "Entries", 3],
     ["RecoJetTrack_absZ0", "Reco Jet abs_Z0", "Entries", 3],
     ["RecoJetTrack_absD0sig", "Reco Jet sigma(abs_D0)", "Entries", 3],
     ["RecoJetTrack_absZ0sig", "Reco Jet sigma(abs_Z0)", "Entries", 3],
     ["RecoJetTrack_D0cov", "Reco Jet cov(D0)", "Entries", 3],
     ["RecoJetTrack_Z0cov", "Reco Jet cov(Z0)", "Entries", 3],

     ["RecoLeadJet_e", "Reco Lead Jet E", "Entries", 3],
     ["RecoLeadJet_pt", "Reco Lead Jet p_{T}", "Entries", 3],
     ["RecoLeadJet_eta", "Reco Lead Jet #eta", "Entries", 3],
     ["RecoLeadJet_phi", "Reco Lead Jet #phi", "Entries", 3],

     ["RecoSecondJet_e", "Reco Secondary Jet E", "Entries", 3],
     ["RecoSecondJet_pt", "Reco Secondary Jet p_{T}", "Entries", 3],
     ["RecoSecondJet_eta", "Reco Secondary Jet #eta", "Entries", 3],
     ["RecoSecondJet_phi", "Reco Secondary Jet #phi", "Entries", 3],

     ["RecoDiJet_e", "Reco Di-Jet E", "Entries", 3],
     ["RecoDiJet_pt", "Reco Di-Jet p_{T}", "Entries", 3],
     ["RecoDiJet_eta", "Reco Di-Jet #eta", "Entries", 3],
     ["RecoDiJet_phi", "Reco Di-Jet #phi", "Entries", 3],


     ["RecoJetDelta_e", "Reco Jet #Delta E", "Entries", 3],
     ["RecoJetDelta_pt", "Reco Jet #Delta p_{T}", "Entries", 3],
     ["RecoJetDelta_phi", "Reco Jet #Delta #phi", "Entries", 3],
     ["RecoJetDelta_eta", "Reco Jet #Delta #eta", "Entries", 3],
     ["RecoJetDelta_R", "Reco Jet #Delta R", "Entries", 3],

     ["GenHNLElectron_e", "GenHNLElectron E", "Entries", 3],
     ["GenHNLElectron_pt", "GenHNLElectron p_{T}", "Entries", 3],
     ["GenHNLElectron_eta", "GenHNLElectron #eta", "Entries", 3],
     ["GenHNLElectron_phi", "GenHNLElectron #phi", "Entries", 3],

     ["LeadJet_HNLELectron_Delta_e", "LeadJet Decay Ele #Delta E", "Entries", 3],
     ["LeadJet_HNLELectron_Delta_pt", "LeadJet Decay Ele #Delta p_{T}", "Entries", 3],
     ["LeadJet_HNLELectron_Delta_eta", "LeadJet Decay Ele #Delta #eta", "Entries", 3],
     ["LeadJet_HNLELectron_Delta_phi", "LeadJet Decay Ele #Delta #phi", "Entries", 3],
     ["LeadJet_HNLELectron_Delta_R", "LeadJet Decay Ele #Delta R", "Entries", 3],

     ["DiJet_HNLElectron_Delta_e", "Di-Jet Decay Ele #Delta E", "Entries", 3],
     ["DiJet_HNLElectron_Delta_pt", "Di-Jet Decay Ele #Delta p_{T}", "Entries", 3],
     ["DiJet_HNLElectron_Delta_phi", "Di-Jet Decay Ele #Delta #phi", "Entries", 3],
     ["DiJet_HNLElectron_Delta_eta", "Di-Jet Decay Ele #Delta #eta", "Entries", 3],
     ["DiJet_HNLElectron_Delta_R", "Di-Jet Decay Ele #Delta R", "Entries", 3],

     ["GenDiJet_invMass", "Gen Di-Jet invariant mass [GeV]", "Entries", 3],
     ["GenDiJetElectron_invMass", "Gen Di-Jet-Electrons invariant mass [GeV]", "Entries", 3],
     ["GenDiJet_electron_invMass", "Gen Di-Jet-electron (-) invariant mass [GeV]", "Entries", 3],
     ["GenDiJet_positron_invMass", "Gen Di-Jet-positron(+) invariant mass [GeV]", "Entries", 3],

     ["GenHNL_DiJet_Delta_theta", "HNL - DiJet #Delta #theta", "Entries", 3],
     ["GenHNL_Electron_Delta_theta", "HNL - Electrons #Delta #theta", "Entries", 3],
     ["GenHNLelectron_Delta_theta", "HNL - electron(-) #Delta #theta", "Entries", 3],
     ["GenHNLpositron_Delta_theta", "HNL - positron (+) #Delta #theta", "Entries", 3],
     ["GenHNLElectron_DiJet_Delta_theta", "Electron - DiJet #Delta #theta", "Entries", 3],
     ["GenHNL_electron_DiJet_Delta_theta", "electron (-) - DiJet #Delta #theta", "Entries", 3],
     ["GenHNL_positron_DiJet_Delta_theta", "positron(+) - DiJet #Delta #theta", "Entries", 3],
     ["GenDiJetElectron_Electron_Delta_theta", "DiJet-Electron-Electron #Delta #theta", "Entries", 3],
     ["GenDiJet_electron_electron_Delta_theta", "DiJet-electron-electron #Delta #theta", "Entries", 3],
     ["GenDiJet_positron_positron_Delta_theta", "DiJet-positron-positron #Delta #theta", "Entries", 3],
     ["GenHNL_positron_e", "positron energy [GeV]", "Entries", 3],
     ["GenHNL_positron_pt", "positron p_{T} [GeV]", "Entries", 3],
     ["GenHNL_electron_e", "electron (-) energy [GeV]", "Entries", 3],
     ["GenHNL_electron_pt", "electron (-) p_{T}", "Entries", 3],

     ["RecoHNLElectron_DiJet_Delta_theta", "Reco DiJet - Electrons #Delta #theta", "Entries", 3],
     ["RecoHNL_electron_DiJet_Delta_theta", "Reco DiJet - electron (-) #Delta #theta", "Entries", 3],
     ["RecoHNL_positron_DiJet_Delta_theta", "Reco DiJet - positron #Delta #theta", "Entries", 3],

     ["RecoDiJetElectron_Electron_Delta_theta", "Reco DiJet_Electron - Electron #Delta #theta", "Entries", 3],
     ["RecoDiJet_electron_electron_Delta_theta", "Reco DiJet_electron (-) - electron #Delta #theta", "Entries", 3],
     ["RecoDiJet_positron_positron_Delta_theta", "Reco DiJet_positron - positron #Delta #theta", "Entries", 3],

     ["RecoDiJet_invMass", "Reco Di-Jet invariant mass [GeV]", "Entries", 3],
     ["RecoDiJetElectron_invMass", "Reco Di-Jet-Electrons invariant mass [GeV]", "Entries", 3],
     ["RecoDiJet_electron_invMass", "Reco Di-Jet-electron (-) invariant mass [GeV]", "Entries", 3],
     ["RecoDiJet_positron_invMass", "Reco Di-Jet-positron(+) invariant mass [GeV]", "Entries", 3],

     
     ["GenHNL_Lxy", "GenHNL_Lxy", "Entries", 3],
     ["GenHNL_Lxyz", "GenHNL_Lxyz", "Entries", 3],

     ["GenLeadJet_phi_e", "LeadJet phi select E [GeV]", "Entries", 3],
     ["GenLeadJet_phi_pt", "LeadJet phi select pt [GeV]", "Entries", 3],
     ["GenSecondJet_phi_e", "SecondJet phi select E [GeV]", "Entries", 3],
     ["GenSecondJet_phi_pt", "SecondJet phi select pt [GeV]", "Entries", 3],
#    ['hnlLT', 'Lifetime [s]', 'Entries'],
#    ['angsepR', 'Reco Cos#theta ee', 'Entries', 5],
#    ['angsep', 'Cos#theta ee', 'Entries', 5],
#    ['et', 'Missing energy [GeV]', 'Entries'],
#    ['eTruthE', 'Electron energy [GeV]', 'Entries'] ,
#    ['eTruthP', 'Positron energy [GeV]', 'Entries'] ,
#    ['eRecoE', 'Reco Electron energy [GeV]', 'Entries'] ,
#    ['eRecoP', 'Reco Positron energy [GeV]', 'Entries'] ,
#    ['etaRE', 'Reco electron #eta', 'Entries'] ,
#    ['etaRP', 'Reco positron #eta', 'Entries'] ,
#    ['phiRE', 'Reco electron #phi', 'Entries'] ,
#    ['phiRP', 'Reco positron #phi', 'Entries'] ,
#    ['deletaR', 'Reco del eta ee', 'Entries'] ,
#    ['delphiR', 'Reco del phi ee', 'Entries'] ,
#    ['delRR', 'Reco del R ee', 'Entries'] ,
#    ['etaE', 'electron #eta', 'Entries'] ,
#    ['etaP', 'positron #eta', 'Entries'] ,
#    ['phiE', 'electron #phi', 'Entries'] ,
#    ['phiP', 'positron #phi', 'Entries'] ,
#    ['deleta', 'del eta ee', 'Entries'] ,
#    ['delphi', 'del phi ee', 'Entries'] ,
#    ['delR', 'del R ee', 'Entries'] ,
#    ['xmRE', 'Reco Px electron [GeV]', 'Entries'] ,
#    ['ymRE', 'Reco Py electron [GeV]', 'Entries'] ,
#    ['zmRE', 'Reco Pz electron [GeV]', 'Entries'] ,
#    ['xmRP', 'Reco Px positron [GeV]', 'Entries'] ,
#    ['ymRP', 'Reco Py positron (GeV', 'Entries'] ,
#    ['zmRP', 'Reco Pz positron [GeV]', 'Entries'] ,
#    ['tmRE', 'Reco pT electron [GeV]', 'Entries'] ,
#    ['tmRP', 'Reco pT positron [GeV]', 'Entries'] ,
#    ['xmTE', 'Px electron [GeV]', 'Entries'] ,
#    ['ymTE', 'Py electron [GeV]', 'Entries'] ,
#    ['zmTE', 'Pz electron [GeV]', 'Entries'] ,
#    ['xmTP', 'Px positron [GeV]', 'Entries'] ,
#    ['ymTP', 'Py positron [GeV]', 'Entries'] ,
#    ['zmTP', 'Pz positron [GeV]', 'Entries'] ,
#    ['tmTE', 'pT electron [GeV]', 'Entries'] ,    
#    ['tmTP', 'pT positron [GeV]', 'Entries'] ,
#   ['dispvrtx', 'displaced vetex [m]', 'Entries']
]

files_list = [
    [input_file_Majorana , 'Majorana ' + HNL_mass + ' semi-leptonic', 'Majorana'],
    [input_file_Dirac , 'Dirac ' + HNL_mass + ' semi-leptonic', 'Dirac']
]

legend_list = [f[1] for f in files_list]
ratio_list = [f[2] for f in files_list]
colors = [609, 856, 410, 801, 629, 879, 602, 921, 622]

def make_plot(h_list, plot_info, legend_list):
   #  print('looking at histogram:', plot_info[0])
    c = ROOT.TCanvas("can"+plot_info[0],"can"+plot_info[0],600,600)
    pad1 = ROOT.TPad("pad1", "pad1",0.0,0.35,1.0,1.0,21)
    pad2 = ROOT.TPad("pad2", "pad2",0.0,0.0,1.0,0.35,22)

    pad1.SetFillColor(0)
    pad1.SetBottomMargin(0.01)
    if log_scale == True : pad1.SetLogy()
    pad1.SetTickx()
    pad1.SetTicky()
    pad1.Draw()

    pad2.SetFillColor(0)
    pad2.SetTopMargin(0.01)
    pad2.SetBottomMargin(0.3)
    pad2.Draw()

    leg = ROOT.TLegend(0.55, 0.7, 0.87, 0.87)
    leg.SetFillStyle(0)
    leg.SetLineWidth(0)

    h_max = 0
    for ih,h in enumerate(h_list):
        leg.AddEntry(h, legend_list[ih])
        if len(plot_info)>3:
            h.Rebin(plot_info[3])
        if h.GetMaximum() > h_max:
            h_max = h.GetMaximum()
        h.Sumw2()

    # Draw in the top panel
    pad1.cd()
    for ih,h in enumerate(h_list):
        h.SetLineColor(colors[ih])
        h.SetLineWidth(3)
        h.GetXaxis().SetTitle(plot_info[1])        
        h.GetYaxis().SetTitle(plot_info[2]) if log_scale == False else h.GetYaxis().SetTitle("log " + plot_info[2])
        h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*1.5)
        h.GetYaxis().SetTitleOffset(0.8)
        h.SetMaximum(1.25*h_max)
        h.Draw('same')
        h.Draw('same E')
    leg.Draw()
    pad1.RedrawAxis()

    # build ratios
    h_ratios = []
    for ih,h in enumerate(h_list):
        if ih == 0:
            h_ratios.append(h.Clone('h_ratio_0'))
            for ibin in range(-1, h.GetNbinsX()+1):
                h_ratios[0].SetBinContent(ibin,1)
        else:
            h_ratios.append(h.Clone('h_ratio_'+str(ih)))
            h_ratios[ih].Divide(h_list[0])

    # draw in the bottom panel
    pad2.cd()
    for ih,h in enumerate(h_ratios):
        h.SetMaximum(1.5)
        h.SetMinimum(0.5)

        h.GetYaxis().SetTitle("Ratio to "+ratio_list[0])
        h.GetYaxis().SetLabelSize(h.GetYaxis().GetLabelSize()*1.6)
        h.GetYaxis().SetLabelOffset(0.01)
        h.GetYaxis().SetTitleSize(h.GetYaxis().GetTitleSize()*1.6)
        h.GetYaxis().SetTitleOffset(0.5)

        h.GetXaxis().SetLabelSize(h.GetXaxis().GetLabelSize()*2.3)
        #h.GetXaxis().SetLabelOffset(0.02)
        h.GetXaxis().SetTitleSize(h.GetXaxis().GetTitleSize()*3)
        h.GetXaxis().SetTitleOffset(1.05)

        h.Draw('hist same')
        if ih>0:
            h.Draw('same E')
    c.SaveAs(output_dir_sel + plot_info[0]+'.png') if log_scale == False else c.SaveAs(output_dir_sel + "log_" + plot_info[0]+'.png')
    return
for plot_info in variables_list:
    h_list = []
    print('looking at histogram:', plot_info[0])
    for ifile,fil in enumerate(files_list):
        f = ROOT.TFile.Open(fil[0])
        h = f.Get(plot_info[0])
        h.SetDirectory(0)
        h_list.append(h)
        f.Close()
    make_plot(h_list, plot_info, legend_list)
