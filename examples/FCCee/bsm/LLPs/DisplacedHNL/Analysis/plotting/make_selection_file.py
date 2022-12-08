import ROOT

selection = "selNone"
#selection = "selJetPtGt20"

input_path = "/afs/cern.ch/user/d/dimoulin/FCCAnalyses_new/FCC-LLP/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/Analysis/outputs/HNL_Majorana_ejj_50GeV_1e-3Ve/output_finalSel/HNL_Majorana_ejj_50GeV_1e-3Ve_"+selection+"_histo.root"
output_file = "selected_hist/histMajorana_ejj_50GeV_1e-3Ve_"+selection+".root"

hist_file = ROOT.TFile.Open(input_path)
#hist_file = ROOT.TFile.Open("../outputs/HNL_Majorana_20GeV_1e-3Ve/HNL_Majorana_20GeV_1e-3Ve_jets_selNone_histo.root")
histSelect = ROOT.TFile.Open(output_file, "RECREATE")

#Object variables
histSelect.WriteObject(hist_file.Get("RecoElectron_pt"), "RecoElectron_pt")
histSelect.WriteObject(hist_file.Get("RecoElectron_phi"), "RecoElectron_phi")
histSelect.WriteObject(hist_file.Get("RecoElectron_theta"), "RecoElectron_theta")
histSelect.WriteObject(hist_file.Get("RecoElectron_e"), "RecoElectron_e")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_e"), "RecoMissingEnergy_e")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_p"), "RecoMissingEnergy_p")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_pt"), "RecoMissingEnergy_pt")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_px"), "RecoMissingEnergy_px")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_py"), "RecoMissingEnergy_py")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_pz"), "RecoMissingEnergy_pz")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_eta"), "RecoMissingEnergy_eta")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_theta"), "RecoMissingEnergy_theta")
histSelect.WriteObject(hist_file.Get("RecoMissingEnergy_phi"), "RecoMissingEnergy_phi")

# Jet variables 
histSelect.WriteObject(hist_file.Get("n_RecoJets"), "n_RecoJets")
histSelect.WriteObject(hist_file.Get("RecoJet_e"), "RecoJet_e")
histSelect.WriteObject(hist_file.Get("RecoJet_p"), "RecoJet_p")
histSelect.WriteObject(hist_file.Get("RecoJet_pt"), "RecoJet_pt")
histSelect.WriteObject(hist_file.Get("RecoJet_pz"), "RecoJet_pz")
histSelect.WriteObject(hist_file.Get("RecoJet_eta"), "RecoJet_eta")
histSelect.WriteObject(hist_file.Get("RecoJet_theta"), "RecoJet_theta")
histSelect.WriteObject(hist_file.Get("RecoJet_phi"), "RecoJet_phi")
histSelect.WriteObject(hist_file.Get("RecoJet_charge"), "RecoJet_charge")
histSelect.WriteObject(hist_file.Get("RecoJetTrack_absD0"), "RecoJetTrack_absD0")
histSelect.WriteObject(hist_file.Get("RecoJetTrack_absZ0"), "RecoJetTrack_absZ0")
histSelect.WriteObject(hist_file.Get("RecoJetTrack_absD0sig"), "RecoJetTrack_absD0sig")
histSelect.WriteObject(hist_file.Get("RecoJetTrack_absZ0sig"), "RecoJetTrack_absZ0sig")
histSelect.WriteObject(hist_file.Get("RecoJetTrack_D0cov"), "RecoJetTrack_D0cov")
histSelect.WriteObject(hist_file.Get("RecoJetTrack_Z0cov"), "RecoJetTrack_Z0cov")

histSelect.WriteObject(hist_file.Get("RecoLeadJet_e"), "RecoLeadJet_e")
histSelect.WriteObject(hist_file.Get("RecoLeadJet_pt"), "RecoLeadJet_pt")
histSelect.WriteObject(hist_file.Get("RecoLeadJet_eta"), "RecoLeadJet_eta")
histSelect.WriteObject(hist_file.Get("RecoLeadJet_phi"), "RecoLeadJet_phi")
histSelect.WriteObject(hist_file.Get("RecoSecondJet_e"), "RecoSecondJet_e")
histSelect.WriteObject(hist_file.Get("RecoSecondJet_pt"), "RecoSecondJet_pt")
histSelect.WriteObject(hist_file.Get("RecoSecondJet_eta"), "RecoSecondJet_eta")
histSelect.WriteObject(hist_file.Get("RecoSecondJet_phi"), "RecoSecondJet_phi")
histSelect.WriteObject(hist_file.Get("RecoJetDelta_e"), "RecoJetDelta_e")
histSelect.WriteObject(hist_file.Get("RecoJetDelta_pt"), "RecoJetDelta_pt")
histSelect.WriteObject(hist_file.Get("RecoJetDelta_eta"), "RecoJetDelta_eta")
histSelect.WriteObject(hist_file.Get("RecoJetDelta_phi"), "RecoJetDelta_phi")
histSelect.WriteObject(hist_file.Get("RecoJetDelta_R"), "RecoJetDelta_R")

histSelect.WriteObject(hist_file.Get("Reco_LeadJet_invMass"), "Reco_LeadJet_invMass")
histSelect.WriteObject(hist_file.Get("LeadJet_HNLELectron_Delta_e"), "LeadJet_HNLELectron_Delta_e")
histSelect.WriteObject(hist_file.Get("LeadJet_HNLELectron_Delta_pt"), "LeadJet_HNLELectron_Delta_pt")
histSelect.WriteObject(hist_file.Get("LeadJet_HNLELectron_Delta_eta"), "LeadJet_HNLELectron_Delta_eta")
histSelect.WriteObject(hist_file.Get("LeadJet_HNLELectron_Delta_phi"), "LeadJet_HNLELectron_Delta_phi")
histSelect.WriteObject(hist_file.Get("LeadJet_HNLELectron_Delta_R"), "LeadJet_HNLELectron_Delta_R")

histSelect.WriteObject(hist_file.Get("GenHNLElectron_e"), "GenHNLElectron_e")
histSelect.WriteObject(hist_file.Get("GenHNLElectron_pt"), "GenHNLElectron_pt")
histSelect.WriteObject(hist_file.Get("GenHNLElectron_eta"), "GenHNLElectron_eta")
histSelect.WriteObject(hist_file.Get("GenHNLElectron_phi"), "GenHNLElectron_phi")

histSelect.WriteObject(hist_file.Get("GenHNL_Lxy"), "GenHNL_Lxy")
histSelect.WriteObject(hist_file.Get("GenHNL_Lxyz"), "GenHNL_Lxyz")

histSelect.WriteObject(hist_file.Get("RecoDiJet_e"), "RecoDiJet_e")
histSelect.WriteObject(hist_file.Get("RecoDiJet_pt"), "RecoDiJet_pt")
histSelect.WriteObject(hist_file.Get("RecoDiJet_eta"), "RecoDiJet_eta")
histSelect.WriteObject(hist_file.Get("RecoDiJet_phi"), "RecoDiJet_phi")

histSelect.WriteObject(hist_file.Get("DiJet_HNLElectron_Delta_e"), "DiJet_HNLElectron_Delta_e")
histSelect.WriteObject(hist_file.Get("DiJet_HNLElectron_Delta_pt"), "DiJet_HNLElectron_Delta_pt")
histSelect.WriteObject(hist_file.Get("DiJet_HNLElectron_Delta_eta"), "DiJet_HNLElectron_Delta_eta")
histSelect.WriteObject(hist_file.Get("DiJet_HNLElectron_Delta_phi"), "DiJet_HNLElectron_Delta_phi")
histSelect.WriteObject(hist_file.Get("DiJet_HNLElectron_Delta_R"), "DiJet_HNLElectron_Delta_R")



#histSelect.WriteObject(hist_file.Get(""), "")


