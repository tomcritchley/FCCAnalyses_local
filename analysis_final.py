#Input directory where the files produced at the stage1 level are
#inputDir  = "/afs/cern.ch/user/d/dimoulin/FCCAnalyses_new/FCC-LLP/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/Analysis/outputs/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_stage1/"
#inputDir = "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_stage1/"
#inputDir = "output_stage1/"
#inputDir = "/eos/user/t/tcritchl/output_stage1_panTest/"
inputDir = "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/HNL_sample_creation/signal_HNLS"
#Input directory where the files produced at the final-selection level are
#outputDir  = "outputs/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_finalSel/"
#outputDir = "/eos/user/d/dimoulin/analysis_outputs/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_finalSel/"
#outputDir = "/eos/user/t/tcritchl/outputs/output_final/testingAll"
outputDir = "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/HNL_sample_creation/signal_HNLS/final_sig"
processList = {
    #run over the full statistics from stage1
    #'HNL_Dirac_ejj_7'':100GeV_1e-3Ve_W2023':{},
    #'p8_ee_Zud_ecm91':{},
    #'p8_ee_Zbb_ecm91':{},

    #'p8_ee_Zcc_ecm91':{},
    #'HNL_ejj_20GeV.root':{},
    #'HNL_ejj_20GeV.root':{},
    #'HNL_ejj_70GeV.root':{},
    'HNL_Dirac_ejj_20GeV_1e-3Ve':{},
    'HNL_Dirac_ejj_50GeV_1e-3Ve':{},
    'HNL_Dirac_ejj_70GeV_1e-3Ve':{},
}


#Link to the dictonary that contains all the cross section information etc...
procDict = "FCCee_procDict_winter2023_IDEA.json"

#Add MySample_p8_ee_ZH_ecm240 as it is not an offical process
procDictAdd={
    #"MySample_p8_ee_ZH_ecm240":{"numberOfEvents": 10000000, "sumOfWeights": 10000000, "crossSection": 0.201868, "kfactor": 1.0, "matchingEfficiency": 1.0}
}

#Number of CPUs to use
nCPUS = 4

#produces ROOT TTrees, default is False
#doTree = True

###Dictionnay of the list of cuts. The key is the name of the selection that will be added to the output file
cutList = {
    "selNone": "n_RecoTracks > -1",
    #"selRecoEleGt0":"n_RecoElectrons > 0", 
    #"selRecoEleEGt11": "n_RecoElectrons > 0 && RecoElectron_lead_e > 11",
    "selMissingEGt25": "RecoMissingEnergy_e[0] > 25",
    "selEleDiJetDRLt27": "RecoElectron_DiJet_delta_R < 2.7",
    "selEleSecondJetDRGt2": "RecoElectron_SecondJet_delta_R > 2",
    "selEleEGt11_MissingEGt25": "RecoElectron_lead_e > 11 && RecoMissingEnergy_e[0] > 25",
    "selEleEGt11_MissingEGt25_EleDiJetDRLt27": "RecoElectron_lead_e > 11 && RecoMissingEnergy_e[0] > 25 && RecoElectron_DiJet_delta_R < 2.7",
    "selEleEGt11_MissingEGt25_EleDiJetDRLt27_EleSecondJetDRGt2": "RecoElectron_lead_e > 11 && RecoMissingEnergy_e[0] > 25 && RecoElectron_DiJet_delta_R < 2.7 && RecoElectron_SecondJet_delta_R > 2",
    


    #"selGenNeutrinoGt0": "n_FSGenNeutrino > 0"
    #"selGenJetEGt20": "GenJet_e[0] > 20",   
    #"selGenMinDrGt04": "GenElRecoJets_ee_kt_minDR[0] > 0.4"
    # "sel1FSGenEle": "n_FSGenElectron>0",
    # "sel1FSGenEle_eeInvMassGt80": "n_FSGenElectron>0 && FSGen_ee_invMass >80",
    # "sel1FSGenNu": "n_FSGenNeutrino>0",
    # "sel2RecoEle": "n_RecoElectrons==2",
    # "sel2RecoEle_vetoes": "n_RecoElectrons==2 && n_RecoMuons==0 && n_RecoPhotons==0 && n_RecoJets_ee_kts==0 && n_RecoPhotons==0",
    # "sel2RecoEle_absD0Gt0p1": "n_RecoElectrons==2 && RecoElectronTrack_absD0[0]>0.1 && RecoElectronTrack_absD0[1]>0.1", #both electrons displaced
    # "sel2RecoEle_chi2Gt0p1": "n_RecoElectrons==2 && RecoDecayVertex.chi2>0.1", #good vertex
    # "sel2RecoEle_chi2Gt0p1_LxyzGt1": "n_RecoElectrons==2 && RecoDecayVertex.chi2>0.1 && Reco_Lxyz>1", #displaced vertex
    # "sel2RecoEle_vetoes_MissingEnergyGt10": "n_RecoElectrons==2 && n_RecoMuons==0 && n_RecoPhotons==0 && n_RecoJets_ee_kts==0 && n_RecoPhotons==0 && RecoMissingEnergy_p[0]>10", #missing energy > 10 GeV
    # "sel2RecoEle_vetoes_absD0Gt0p5": "n_RecoElectrons==2 && n_RecoMuons==0 && n_RecoPhotons==0 && n_RecoJets_ee_kts==0 && n_RecoPhotons==0 && RecoElectronTrack_absD0[0]>0.5 && RecoElectronTrack_absD0[1]>0.5", #both electrons displaced
    #"sel2RecoEle_vetoes_MissingEnergyGt10_absD0Gt0p5": "n_RecoElectrons==2 && n_RecoMuons==0 && n_RecoPhotons==0 && n_RecoJets_ee_kts==0 && n_RecoPhotons==0 && RecoMissingEnergy_p[0]>10 && RecoElectronTrack_absD0[0]>0.5 && RecoElectronTrack_absD0[1]>0.5", #both electrons displaced
    # "sel2RecoEle_vetoes_MissingEnergyGt10_chi2Gt1_LxyzGt5": "n_RecoElectrons==2 && n_RecoMuons==0 && n_RecoPhotons==0 && n_RecoJets_ee_kts==0 && n_RecoPhotons==0 && RecoMissingEnergy_p[0]>10 && RecoDecayVertex.chi2>1 && Reco_Lxyz>5", #displaced vertex
}

###Dictionary for the ouput variable/hitograms. The key is the name of the variable in the output files. "name" is the name of the variable in the input file, "title" is the x-axis label of the histogram, "bin" the number of bins of the histogram, "xmin" the minimum x-axis value and "xmax" the maximum x-axis value.
histoList = {

    #gen variables
    "n_FSGenElectron":                   {"name":"n_FSGenElectron",                  "title":"Number of final state gen electrons",        "bin":5,"xmin":-0.5 ,"xmax":4.5},

    "FSGenElectron_e":                   {"name":"FSGenElectron_e",                  "title":"Final state gen electrons energy [GeV]",     "bin":100,"xmin":0 ,"xmax":50},
    "FSGenElectron_p":                   {"name":"FSGenElectron_p",                  "title":"Final state gen electrons p [GeV]",          "bin":100,"xmin":0 ,"xmax":50},
    "FSGenElectron_pt":                  {"name":"FSGenElectron_pt",                 "title":"Final state gen electrons p_{T} [GeV]",      "bin":100,"xmin":0 ,"xmax":50},
    "FSGenElectron_pz":                  {"name":"FSGenElectron_pz",                 "title":"Final state gen electrons p_{z} [GeV]",      "bin":100,"xmin":0 ,"xmax":50},
    "FSGenElectron_eta":                 {"name":"FSGenElectron_eta",                "title":"Final state gen electrons #eta",             "bin":60, "xmin":-3,"xmax":3},
    "FSGenElectron_theta":               {"name":"FSGenElectron_theta",              "title":"Final state gen electrons #theta",           "bin":64, "xmin":0,"xmax":3.2},
    "FSGenElectron_phi":                 {"name":"FSGenElectron_phi",                "title":"Final state gen electrons #phi",             "bin":64, "xmin":-3.2,"xmax":3.2},
    "FSGenElectron_charge":              {"name":"FSGenElectron_charge",             "title":"Final state gen electrons charge",           "bin":3, "xmin":-1.5,"xmax":1.5},

    "FSGenElectron_vertex_x": {"name":"FSGenElectron_vertex_x", "title":"Final state gen e^{#font[122]{\55}} production vertex x [mm]",      "bin":100,"xmin":-1000 ,"xmax":1000},
    "FSGenElectron_vertex_y": {"name":"FSGenElectron_vertex_y", "title":"Final state gen e^{#font[122]{\55}} production vertex y [mm]",      "bin":100,"xmin":-1000 ,"xmax":1000},
    "FSGenElectron_vertex_z": {"name":"FSGenElectron_vertex_z", "title":"Final state gen e^{#font[122]{\55}} production vertex z [mm]",      "bin":100,"xmin":-1000 ,"xmax":1000},

    "FSGenElectron_vertex_x_prompt": {"name":"FSGenElectron_vertex_x", "title":"Final state gen e^{#font[122]{\55}} production vertex x [mm]",      "bin":100,"xmin":-0.01 ,"xmax":0.01},
    "FSGenElectron_vertex_y_prompt": {"name":"FSGenElectron_vertex_y", "title":"Final state gen e^{#font[122]{\55}} production vertex y [mm]",      "bin":100,"xmin":-0.01 ,"xmax":0.01},
    "FSGenElectron_vertex_z_prompt": {"name":"FSGenElectron_vertex_z", "title":"Final state gen e^{#font[122]{\55}} production vertex z [mm]",      "bin":100,"xmin":-0.01 ,"xmax":0.01},

    "FSGen_Lxy":      {"name":"FSGen_Lxy",      "title":"Gen L_{xy} [mm]",     "bin":100,"xmin":0 ,"xmax":1000},
    "FSGen_Lxyz":     {"name":"FSGen_Lxyz",     "title":"Gen L_{xyz} [mm]",    "bin":100,"xmin":0 ,"xmax":1000},
    "FSGen_Lxyz_prompt":     {"name":"FSGen_Lxyz",     "title":"Gen L_{xyz} [mm]",    "bin":100,"xmin":0 ,"xmax":10},

    "n_FSGenNeutrino":                   {"name":"n_FSGenNeutrino",                  "title":"Number of final state gen neutrinos",        "bin":5,"xmin":-0.5 ,"xmax":4.5},

    "FSGenNeutrino_e":                   {"name":"FSGenNeutrino_e",                  "title":"Final state gen neutrinos energy [GeV]",     "bin":100,"xmin":0 ,"xmax":50},
    "FSGenNeutrino_p":                   {"name":"FSGenNeutrino_p",                  "title":"Final state gen neutrinos p [GeV]",          "bin":100,"xmin":0 ,"xmax":50},
    "FSGenNeutrino_pt":                  {"name":"FSGenNeutrino_pt",                 "title":"Final state gen neutrinos p_{T} [GeV]",      "bin":100,"xmin":0 ,"xmax":50},
    "FSGenNeutrino_pz":                  {"name":"FSGenNeutrino_pz",                 "title":"Final state gen neutrinos p_{z} [GeV]",      "bin":100,"xmin":0 ,"xmax":50},
    "FSGenNeutrino_eta":                 {"name":"FSGenNeutrino_eta",                "title":"Final state gen neutrinos #eta",             "bin":60, "xmin":-3,"xmax":3},
    "FSGenNeutrino_theta":               {"name":"FSGenNeutrino_theta",              "title":"Final state gen neutrinos #theta",           "bin":64, "xmin":0,"xmax":3.2},
    "FSGenNeutrino_phi":                 {"name":"FSGenNeutrino_phi",                "title":"Final state gen neutrinos #phi",             "bin":64, "xmin":-3.2,"xmax":3.2},
    "FSGenNeutrino_charge":              {"name":"FSGenNeutrino_charge",             "title":"Final state gen neutrinos charge",           "bin":3, "xmin":-1.5,"xmax":1.5},

    #"n_GenJets":       {"name":"n_GenJets",      "title":"Number of gen jets per event",         "bin":11,"xmin":-0.5 ,"xmax":20.5},

    "FSGen_ee_invMass":   {"name":"FSGen_ee_invMass",   "title":"Gen m_{ee} [GeV]",           "bin":100,"xmin":0, "xmax":100},
    "FSGen_eenu_invMass": {"name":"FSGen_eenu_invMass", "title":"Gen m_{ee#nu} [GeV]",           "bin":100,"xmin":0, "xmax":100},

    "n_FSGenPhoton":                   {"name":"n_FSGenPhoton",                  "title":"Number of final state gen photons",          "bin":10,"xmin":-0.5 ,"xmax":9.5},
    "FSGenPhoton_e":                   {"name":"FSGenPhoton_e",                  "title":"Final state gen photons energy [GeV]",       "bin":100,"xmin":0 ,"xmax":50},
    "FSGenPhoton_p":                   {"name":"FSGenPhoton_p",                  "title":"Final state gen photons p [GeV]",            "bin":100,"xmin":0 ,"xmax":50},
    "FSGenPhoton_pt":                  {"name":"FSGenPhoton_pt",                 "title":"Final state gen photons p_{T} [GeV]",        "bin":100,"xmin":0 ,"xmax":50},
    "FSGenPhoton_pz":                  {"name":"FSGenPhoton_pz",                 "title":"Final state gen photons p_{z} [GeV]",        "bin":100,"xmin":0 ,"xmax":50},
    "FSGenPhoton_eta":                 {"name":"FSGenPhoton_eta",                "title":"Final state gen photons #eta",               "bin":60, "xmin":-3,"xmax":3},
    "FSGenPhoton_theta":               {"name":"FSGenPhoton_theta",              "title":"Final state gen photons #theta",             "bin":64, "xmin":0,"xmax":3.2},
    "FSGenPhoton_phi":                 {"name":"FSGenPhoton_phi",                "title":"Final state gen photons #phi",               "bin":64, "xmin":-3.2,"xmax":3.2},
    "FSGenPhoton_charge":              {"name":"FSGenPhoton_charge",             "title":"Final state gen photons charge",             "bin":3, "xmin":-1.5,"xmax":1.5},
    
    #"GenHNL_electron_e":        {"name":"GenHNL_electron_e",        "title":"Decay electron(-) energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    #"GenHNL_electron_pt":        {"name":"GenHNL_electron_pt",        "title":"Decay electron(-) pt [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    
    #"GenHNL_positron_e":        {"name":"GenHNL_positron_e",        "title":"Decay positron(+) energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    #"GenHNL_positron_pt":        {"name":"GenHNL_positron_pt",        "title":"Decay positron(+) pt [GeV]", "bin":100,"xmin":0 ,"xmax":50},

    #"GenHNLElectron_e":        {"name":"GenHNLElectron_e",        "title":"(Gen)Decay electron energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    #"GenHNLElectron_pt":        {"name":"GenHNLElectron_pt",        "title":"(Gen)Decay electron p_{T} [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    #"GenHNLElectron_eta":        {"name":"GenHNLElectron_eta",        "title":"(Gen)Decay electron #eta", "bin":60,"xmin":-3 ,"xmax":3},
    #"GenHNLElectron_phi":        {"name":"GenHNLElectron_phi",        "title":"(Gen)Decay electron #phi", "bin":64,"xmin":-3.2 ,"xmax":3.2},

    #"GenDiJetElectron_e":        {"name":"GenDiJetElectron_e",        "title":"(Gen)DiJet + Electron energy [GeV]", "bin":100,"xmin":0 ,"xmax":100},
    #"GenDiJetElectron_pt":        {"name":"GenDiJetElectron_pt",        "title":"(Gen)DiJet + Electron p_{T} [GeV]", "bin":100,"xmin":0 ,"xmax":100},
    #"GenDiJetElectron_eta":        {"name":"GenDiJetElectron_eta",        "title":"(Gen)DiJet + Electron #eta", "bin":60,"xmin":-3 ,"xmax":3},
    #"GenDiJetElectron_phi":        {"name":"GenDiJetElectron_phi",        "title":"(Gen)DiJet + Electron #phi", "bin":64,"xmin":-3.2 ,"xmax":3.2},

    #"GenElRecoJets_ee_kt_minDR":        {"name":"GenElRecoJets_ee_kt_minDR",        "title":"Gen Electron - Reco Jet #Delta R", "bin":150,"xmin":0 ,"xmax":5},


    #reco variables
    "n_RecoTracks":                    {"name":"n_RecoTracks",                   "title":"Total number of reco tracks",           "bin":5,"xmin":-0.5 ,"xmax":4.5},
    "n_RecoPhotons":    {"name":"n_RecoPhotons",   "title":"Total number of reco photons",      "bin":5,"xmin":-0.5 ,"xmax":4.5},
    "n_RecoElectrons":  {"name":"n_RecoElectrons", "title":"Total number of reco electrons",    "bin":5,"xmin":-0.5 ,"xmax":4.5},
    "n_RecoMuons":      {"name":"n_RecoMuons",     "title":"Total number of reco muons",        "bin":5,"xmin":-0.5 ,"xmax":4.5},

    "RecoElRecoJets_ee_kt_minDR":        {"name":"RecoElRecoJets_ee_kt_minDR",        "title":"Reco Electron - Reco Jet #Delta R", "bin":150,"xmin":0 ,"xmax":5},
    "RecoJets_ee_kt_n":       {"name":"RecoJets_ee_kt_n",      "title":"Total number of reco jets",         "bin":11,"xmin":-0.5 ,"xmax":10.5},
    "RecoJets_ee_kt_e":        {"name":"RecoJets_ee_kt_e",        "title":"Reco jet energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoJets_ee_kt_pt":       {"name":"RecoJets_ee_kt_pt",       "title":"Reco jet p_{T} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoJets_ee_kt_eta":      {"name":"RecoJets_ee_kt_eta",      "title":"Reco jet #eta",         "bin":60, "xmin":-3,"xmax":3},
    "RecoJets_ee_kt_theta":    {"name":"RecoJets_ee_kt_theta",    "title":"Reco jet #theta",       "bin":64, "xmin":0,"xmax":3.2},
    "RecoJets_ee_kt_phi":      {"name":"RecoJets_ee_kt_phi",      "title":"Reco jet #phi",         "bin":64, "xmin":-3.2,"xmax":3.2},

   "GenJets_ee_kt_e":      {"name":"GenJets_ee_kt_e",     "title":"Gen jet energy [GeV]",      "bin":100,"xmin":0, "xmax":100},

    #"GenJet_e":      {"name":"GenJet_e",     "title":"Gen jet energy [GeV]",      "bin":100,"xmin":0, "xmax":5},  
    #"GenLeadJet_e":      {"name":"GenLeadJet_e",     "title":"Gen Leading jet energy [GeV]",      "bin":100,"xmin":0, "xmax":50},   
    #"GenLeadJet_pt":      {"name":"GenLeadJet_pt",     "title":"Gen Leading jet pt [GeV]",      "bin":100,"xmin":0, "xmax":50},
    #"GenLeadJet_eta":      {"name":"GenLeadJet_eta",     "title":"Gen Leading jet eta [GeV]",      "bin":100,"xmin":-3.2, "xmax":3.2},
    #"GenLeadJet_phi":      {"name":"GenLeadJet_phi",     "title":"Gen Leading jet phi [GeV]",      "bin":100,"xmin":-3.2, "xmax":3.2},

    #"GenSecondJet_e":      {"name":"GenSecondJet_e",     "title":"Gen Second leading jet energy [GeV]",      "bin":100,"xmin":0, "xmax":50},
    #"GenSecondJet_pt":      {"name":"GenSecondJet_pt",     "title":"Gen Second leading jet pt [GeV]",      "bin":100,"xmin":0, "xmax":50},
    #"GenSecondJet_eta":      {"name":"GenSecondJet_eta",     "title":"Gen Second leading jet eta [GeV]",      "bin":100,"xmin":-3.2, "xmax":3.2},
    #"GenSecondJet_phi":      {"name":"GenSecondJet_phi",     "title":"Gen Second leading jet phi [GeV]",      "bin":100,"xmin":-3.2, "xmax":3.2},


    #"GenLeadJet_phi_e":      {"name":"GenLeadJet_phi_e",     "title":"Gen Leading jet energy phi_sel [GeV]",      "bin":100,"xmin":0, "xmax":50},
    #"GenLeadJet_phi_pt":      {"name":"GenLeadJet_phi_pt",     "title":"Gen Leading jet pt phi_sel [GeV]",      "bin":100,"xmin":0, "xmax":50},
    #"GenSecondJet_phi_e":      {"name":"GenSecondJet_phi_e",     "title":"Gen Second jet energy phi_sel [GeV]",      "bin":100,"xmin":0, "xmax":50},
    #"GenSecondJet_phi_pt":      {"name":"GenSecondJet_phi_pt",     "title":"Gen Second jet pt phi_sel [GeV]",      "bin":100,"xmin":0, "xmax":50},

    #"GenDiJet_e":        {"name":"GenDiJet_e",        "title":"Gen Di-jet energy [GeV]", "bin":100,"xmin":0 ,"xmax":70},
    #"GenDiJet_pt":        {"name":"GenDiJet_pt",        "title":"Gen Di-jet p_{T} [GeV]", "bin":100,"xmin":0 ,"xmax":70},
    #"GenDiJet_eta":        {"name":"GenDiJet_eta",        "title":"Gen Di-jet #eta", "bin":64,"xmin":-3.2 ,"xmax":3.2},
    #"GenDiJet_phi":        {"name":"GenDiJet_phi",        "title":"Gen Di-jet #phi", "bin":64,"xmin":-3.2 ,"xmax":3.2},


    #"GenDiJet_invMass":   {"name":"GenDiJet_invMass",   "title":"Gen DiJet mass [GeV]",           "bin":100,"xmin":0, "xmax":100},

    #"GenDiJetElectron_invMass":   {"name":"GenDiJetElectron_invMass",   "title":"Gen DiJet - Electron mass [GeV]",           "bin":100,"xmin":0, "xmax":100},
    #"GenDiJet_electron_invMass":   {"name":"GenDiJet_electron_invMass",   "title":"Gen DiJet - electron mass [GeV]",           "bin":100,"xmin":0, "xmax":100},
    #"GenDiJet_positron_invMass":   {"name":"GenDiJet_positron_invMass",   "title":"Gen DiJet - positron mass [GeV]",           "bin":100,"xmin":0, "xmax":100},

    "RecoLeadJet_e":      {"name":"RecoLeadJet_e",     "title":"Reco Leading jet energy [GeV]",      "bin":100,"xmin":0, "xmax":50},
    "RecoLeadJet_e_scaled":      {"name":"RecoLeadJet_e",     "title":"Reco Leading jet energy_scaled [GeV]",      "bin":100,"xmin":0, "xmax":1},
    "RecoLeadJet_pt":      {"name":"RecoLeadJet_pt",     "title":"Reco Leading jet p_{T} [GeV]",      "bin":100,"xmin":0, "xmax":50},
    "RecoLeadJet_eta":      {"name":"RecoLeadJet_eta",     "title":"Reco Leading jet #eta",      "bin":64,"xmin":-3.2, "xmax":3.2},
    "RecoLeadJet_phi":      {"name":"RecoLeadJet_phi",     "title":"Reco Leading jet #phi",      "bin":64,"xmin":0, "xmax":6.4},
    "RecoSecondJet_e":      {"name":"RecoSecondJet_e",     "title":"Reco Secondary jet energy [GeV]",      "bin":100,"xmin":0, "xmax":50},
    "RecoSecondJet_pt":      {"name":"RecoSecondJet_pt",     "title":"Reco Secondary jet p_{T} [GeV]",      "bin":100,"xmin":0, "xmax":50},
    "RecoSecondJet_eta":      {"name":"RecoSecondJet_eta",     "title":"Reco Secondary jet #eta",      "bin":64,"xmin":-3.2, "xmax":3.2},
    "RecoSecondJet_phi":      {"name":"RecoSecondJet_phi",     "title":"Reco Secondary jet #phi",      "bin":64,"xmin":0, "xmax":6.4},

    "RecoDiJet_e":        {"name":"RecoDiJet_e",        "title":"Reco di-jet energy [GeV]", "bin":100,"xmin":0 ,"xmax":70},
    "RecoDiJet_pt":        {"name":"RecoDiJet_pt",        "title":"Reco di-jet pt [GeV]", "bin":100,"xmin":0 ,"xmax":70},
    "RecoDiJet_eta":        {"name":"RecoDiJet_eta",        "title":"Reco di-jet eta", "bin":64,"xmin":-3.2 ,"xmax":3.2},
    "RecoDiJet_phi":        {"name":"RecoDiJet_phi",        "title":"Reco di-jet phi]", "bin":64,"xmin":-3.2 ,"xmax":3.2},


    "RecoJets_ee_ktDelta_e":      {"name":"RecoJets_ee_ktDelta_e",     "title":"Reco Jet #Delta E [GeV]",      "bin":100,"xmin":0, "xmax":50},
    "RecoJets_ee_ktDelta_pt":      {"name":"RecoJets_ee_ktDelta_pt",     "title":"Reco Jet #Delta p_{T} [GeV]",      "bin":100,"xmin":0, "xmax":50},
    "RecoJets_ee_ktDelta_phi":      {"name":"RecoJets_ee_ktDelta_phi",     "title":"Reco Jet #Delta #phi",      "bin":64,"xmin":-4, "xmax":4},
    "RecoJets_ee_ktDelta_eta":      {"name":"RecoJets_ee_ktDelta_eta",     "title":"Reco Jet #Delta #eta",      "bin":64,"xmin":-7, "xmax":7},
    "RecoJets_ee_ktDelta_R":      {"name":"RecoJets_ee_ktDelta_R",     "title":"Reco Jet #Delta R",      "bin":100,"xmin":0, "xmax":10},

    "RecoDiJetElectron_e":      {"name":"RecoDiJetElectron_e",     "title":"Reco DiJet Electron E [GeV]",      "bin":100,"xmin":0, "xmax":100},
    "RecoDiJetElectron_pt":        {"name":"RecoDiJetElectron_pt",        "title":"(Reco)DiJet + Electron p_{T} [GeV]", "bin":100,"xmin":0 ,"xmax":100},
    "RecoDiJetElectron_eta":        {"name":"RecoDiJetElectron_eta",        "title":"(Reco)DiJet + Electron #eta", "bin":60,"xmin":-3 ,"xmax":3},
    "RecoDiJetElectron_phi":        {"name":"RecoDiJetElectron_phi",        "title":"(Reco)DiJet + Electron #phi", "bin":64,"xmin":-3.2 ,"xmax":3.2},
 

    "RecoDiJetElectron_px":      {"name":"RecoDiJetElectron_px",     "title":"Reco DiJet Electron px [GeV]",      "bin":100,"xmin":0, "xmax":70},
    "RecoDiJetElectron_py":      {"name":"RecoDiJetElectron_py",     "title":"Reco DiJet Electron py [GeV]",      "bin":100,"xmin":0, "xmax":50},
    "RecoDiJetElectron_pz":      {"name":"RecoDiJetElectron_pz",     "title":"Reco DiJet Electron pz [GeV]",      "bin":100,"xmin":0, "xmax":50},

    
    "RecoDiJet_invMass":   {"name":"RecoDiJet_invMass",   "title":"Reco DiJet mass [GeV]",           "bin":100,"xmin":0, "xmax":100},
    "RecoDiJetElectron_invMass":   {"name":"RecoDiJetElectron_invMass",   "title":"Reco DiJet-Electrons mass [GeV]",           "bin":100,"xmin":0, "xmax":100},
    "RecoDiJet_electron_invMass":   {"name":"RecoDiJet_electron_invMass",   "title":"Reco DiJet-electron (-) mass [GeV]",           "bin":100,"xmin":0, "xmax":100},
    "RecoDiJet_positron_invMass":   {"name":"RecoDiJet_positron_invMass",   "title":"Reco DiJet-positron (-) mass [GeV]",           "bin":100,"xmin":0, "xmax":100},

    



    "RecoElectron_LeadJet_delta_R":      {"name":"RecoElectron_LeadJet_delta_R",     "title":"RecoElectron - LeadJet #Delta R",      "bin":100,"xmin":0, "xmax":10},
    "RecoElectron_SecondJet_delta_R":      {"name":"RecoElectron_SecondJet_delta_R",     "title":"RecoElectron - SecondJet #Delta R",      "bin":100,"xmin":0, "xmax":10},
    "RecoElectron_DiJet_delta_R":      {"name":"RecoElectron_DiJet_delta_R",     "title":"RecoElectron - DiJet #Delta R",      "bin":100,"xmin":0, "xmax":10},
    "RecoDiJet_delta_R":      {"name":"RecoDiJet_delta_R",     "title":"RecoDiJet #Delta R",      "bin":100,"xmin":0, "xmax":10},


    #"LeadJet_HNLELectron_Delta_e":      {"name":"LeadJet_HNLELectron_Delta_e",     "title":"RecoJets_ee_kt DecayEle #Delta E [GeV]",      "bin":100,"xmin":-50, "xmax":50},
    #"LeadJet_HNLELectron_Delta_pt":      {"name":"LeadJet_HNLELectron_Delta_pt",     "title":"RecoJets_ee_kt DecayEle #Delta p_{T} [GeV]",      "bin":100,"xmin":-50, "xmax":50},
    #"LeadJet_HNLELectron_Delta_phi":      {"name":"LeadJet_HNLELectron_Delta_phi",     "title":"RecoJets_ee_kt DecayEle #Delta #phi",      "bin":64,"xmin":-4, "xmax":4},
    #"LeadJet_HNLELectron_Delta_eta":      {"name":"LeadJet_HNLELectron_Delta_eta",     "title":"RecoJets_ee_kt DecayEle #Delta #eta",      "bin":64,"xmin":-7, "xmax":7},
    #"LeadJet_HNLELectron_Delta_R":      {"name":"LeadJet_HNLELectron_Delta_R",     "title":"RecoJets_ee_kt DecayEle #Delta R",      "bin":100,"xmin":0, "xmax":10},


    #"DiJet_HNLElectron_Delta_e":      {"name":"DiJet_HNLElectron_Delta_e",     "title":"DiJet DecayEle #Delta E [GeV]",      "bin":100,"xmin":-50, "xmax":50},
    #"DiJet_HNLElectron_Delta_pt":      {"name":"DiJet_HNLElectron_Delta_pt",     "title":"DiJet DecayEle #Delta p_{T} [GeV]",      "bin":100,"xmin":-50, "xmax":50},
    #"DiJet_HNLElectron_Delta_phi":      {"name":"DiJet_HNLElectron_Delta_phi",     "title":"DiJet DecayEle #Delta #phi",      "bin":64,"xmin":-4, "xmax":4},
    #"DiJet_HNLElectron_Delta_eta":      {"name":"DiJet_HNLElectron_Delta_eta",     "title":"DiJet DecayEle #Delta #eta",      "bin":64,"xmin":-7, "xmax":7},
    #"DiJet_HNLElectron_Delta_R":      {"name":"DiJet_HNLElectron_Delta_R",     "title":"DiJet DecayEle #Delta R",      "bin":100,"xmin":0, "xmax":10},

    #Diff in theta 
    #"GenHNL_DiJet_Delta_theta":      {"name":"GenHNL_DiJet_Delta_theta",     "title":"HNL - DiJet #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    #"GenHNL_Electron_Delta_theta":      {"name":"GenHNL_Electron_Delta_theta",     "title":"HNL - Electrons #Delta #theta",      "bin":64,"xmin":-4, "xmax":4}, 
    #"GenHNL_Electron_cos_theta":      {"name":"GenHNL_Electron_cos_theta",     "title":"HNL - Electrons cos #Delta#theta",      "bin":64,"xmin":-1, "xmax":1},
    #"GenHNLelectron_Delta_theta":      {"name":"GenHNLelectron_Delta_theta",     "title":"HNL - electron #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    #"GenHNLpositron_Delta_theta":      {"name":"GenHNLpositron_Delta_theta",     "title":"HNL - positron #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    #"GenHNLElectron_DiJet_Delta_theta":      {"name":"GenHNLElectron_DiJet_Delta_theta",     "title":"Electrons - DiJet #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    #"GenHNL_electron_DiJet_Delta_theta":      {"name":"GenHNL_electron_DiJet_Delta_theta",     "title":"electron - Dijet #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    #"GenHNL_positron_DiJet_Delta_theta":      {"name":"GenHNL_positron_DiJet_Delta_theta",     "title":"positron - DiJet #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    #"GenDiJetElectron_Electron_Delta_theta":      {"name":"GenDiJetElectron_Electron_Delta_theta",     "title":"DiJetElectron - Electron #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    #"GenDiJet_electron_electron_Delta_theta":      {"name":"GenDiJet_electron_electron_Delta_theta",     "title":"DiJet-electron-electron #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    #"GenDiJet_positron_positron_Delta_theta":      {"name":"GenDiJet_positron_positron_Delta_theta",     "title":"DiJet-positron-positron #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    

    "RecoHNLElectron_DiJet_Delta_theta":      {"name":"RecoHNLElectron_DiJet_Delta_theta",     "title":"HNLElectron - DiJet #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    "RecoHNL_electron_DiJet_Delta_theta":      {"name":"RecoHNL_electron_DiJet_Delta_theta",     "title":"HNLelectron (-) - DiJet #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    "RecoHNL_positron_DiJet_Delta_theta":      {"name":"RecoHNL_positron_DiJet_Delta_theta",     "title":"HNLpositron - DiJet #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},

    "RecoDiJetElectron_Electron_Delta_theta":      {"name":"RecoDiJetElectron_Electron_Delta_theta",     "title":"RecoHNL - Electron #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    "RecoDiJet_electron_electron_Delta_theta":      {"name":"RecoDiJet_electron_electron_Delta_theta",     "title":"RecoHNL - electron (-) #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},
    "RecoDiJet_positron_positron_Delta_theta":      {"name":"RecoDiJet_positron_positron_Delta_theta",     "title":"RecoHNL - positron #Delta #theta",      "bin":64,"xmin":-4, "xmax":4},

    #"GenHNL_Lxy":                     {"name":"GenHNL_Lxy",                    "title":"Gen L_{xy} [mm]",     "bin":50,"xmin":0 ,"xmax":0.01},
    #"GenHNL_Lxyz":                     {"name":"GenHNL_Lxyz",                    "title":"Gen L_{xyz} [mm]",     "bin":50,"xmin":0 ,"xmax":0.01},


    "RecoElectron_e":        {"name":"RecoElectron_e",        "title":"Reco electron energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoElectron_p":        {"name":"RecoElectron_p",        "title":"Reco electron p [GeV]",      "bin":100,"xmin":0 ,"xmax":50},
    "RecoElectron_pt":       {"name":"RecoElectron_pt",       "title":"Reco electron p_{T} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoElectron_pz":       {"name":"RecoElectron_pz",       "title":"Reco electron p_{z} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoElectron_eta":      {"name":"RecoElectron_eta",      "title":"Reco electron #eta",         "bin":60, "xmin":-3,"xmax":3},
    "RecoElectron_theta":    {"name":"RecoElectron_theta",    "title":"Reco electron #theta",       "bin":64, "xmin":0,"xmax":3.2},
    "RecoElectron_phi":      {"name":"RecoElectron_phi",      "title":"Reco electron #phi",         "bin":64, "xmin":-3.2,"xmax":3.2},
    "RecoElectron_charge":   {"name":"RecoElectron_charge",   "title":"Reco electron charge",       "bin":3, "xmin":-1.5,"xmax":1.5},

    "RecoElectron_lead_e":        {"name":"RecoElectron_lead_e",        "title":"Reco Electron (from HNL) energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoElectron_lead_pt":        {"name":"RecoElectron_lead_pt",        "title":"Reco Electron (from HNL) pt [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoElectron_lead_eta":        {"name":"RecoElectron_lead_eta",        "title":"Reco Electron (from HNL) eta [GeV]", "bin":100,"xmin":-3.2,"xmax":3.2},
    "RecoElectron_lead_phi":        {"name":"RecoElectron_lead_phi",        "title":"Reco Electron (from HNL) phi [GeV]", "bin":100,"xmin":-3.2 ,"xmax":3.2},


    "RecoHNL_electron_e":        {"name":"RecoHNL_electron_e",        "title":"Reco electron (-) energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoHNL_electron_pt":       {"name":"RecoHNL_electron_pt",       "title":"Reco electron (-) p_{T} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoHNL_electron_eta":      {"name":"RecoHNL_electron_eta",      "title":"Reco electron (-) #eta",         "bin":60, "xmin":-3,"xmax":3},
    "RecoHNL_electron_theta":    {"name":"RecoHNL_electron_theta",    "title":"Reco electron (-) #theta",       "bin":64, "xmin":0,"xmax":3.2},
    "RecoHNL_electron_phi":      {"name":"RecoHNL_electron_phi",      "title":"Reco electron (-) #phi",         "bin":64, "xmin":-3.2,"xmax":3.2},

    "RecoHNL_positron_e":        {"name":"RecoHNL_positron_e",        "title":"Reco positron (+) energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoHNL_positron_pt":       {"name":"RecoHNL_positron_pt",       "title":"Reco positron (+) p_{T} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoHNL_positron_eta":      {"name":"RecoHNL_positron_eta",      "title":"Reco positron (+) #eta",         "bin":60, "xmin":-3,"xmax":3},
    "RecoHNL_positron_theta":    {"name":"RecoHNL_positron_theta",    "title":"Reco positron (+) #theta",       "bin":64, "xmin":0,"xmax":3.2},
    "RecoHNL_positron_phi":      {"name":"RecoHNL_positron_phi",      "title":"Reco positron (+) #phi",         "bin":64, "xmin":-3.2,"xmax":3.2},


    "RecoElectronTrack_absD0":             {"name":"RecoElectronTrack_absD0",     "title":"Reco positron tracks |d_{0}| [mm]",      "bin":100,"xmin":0, "xmax":700},
    "RecoElectronTrack_absD0_med":         {"name":"RecoElectronTrack_absD0",     "title":"Reco positron tracks |d_{0}| [mm]",      "bin":100,"xmin":0, "xmax":10},
    "RecoElectronTrack_absD0_prompt":      {"name":"RecoElectronTrack_absD0",     "title":"Reco positron tracks |d_{0}| [mm]",      "bin":100,"xmin":0, "xmax":1},
    "RecoElectronTrack_absZ0":             {"name":"RecoElectronTrack_absZ0",     "title":"Reco positron tracks |z_{0}| [mm]",      "bin":100,"xmin":0, "xmax":700},
    "RecoElectronTrack_absZ0_prompt":      {"name":"RecoElectronTrack_absZ0",     "title":"Reco positron tracks |z_{0}| [mm]",      "bin":100,"xmin":0, "xmax":1},
    "RecoElectronTrack_absD0sig":          {"name":"RecoElectronTrack_absD0sig",  "title":"Reco positron tracks |d_{0} significance|",      "bin":100,"xmin":0, "xmax":50},
    "RecoElectronTrack_absD0sig_prompt":   {"name":"RecoElectronTrack_absD0sig",  "title":"Reco positron tracks |d_{0} significance|",      "bin":100,"xmin":0, "xmax":5},
    "RecoElectronTrack_absZ0sig":          {"name":"RecoElectronTrack_absZ0sig",  "title":"Reco electron tracks |z_{0} significance|",      "bin":100,"xmin":0, "xmax":50},
    "RecoElectronTrack_absZ0sig_prompt":   {"name":"RecoElectronTrack_absZ0sig",  "title":"Reco electron tracks |z_{0} significance|",      "bin":100,"xmin":0, "xmax":5},
    "RecoElectronTrack_D0cov":      {"name":"RecoElectronTrack_D0cov",     "title":"Reco electron tracks d_{0} #sigma^{2}",      "bin":100,"xmin":0, "xmax":0.5},
    "RecoElectronTrack_Z0cov":      {"name":"RecoElectronTrack_Z0cov",     "title":"Reco electron tracks z_{0} #sigma^{2}",      "bin":100,"xmin":0, "xmax":0.5},

    "Reco_DecayVertex_x":           {"name":"RecoDecayVertex.position.x",  "title":"Reco decay vertex x [mm]",            "bin":100,"xmin":-1000 ,"xmax":1000},
    "Reco_DecayVertex_y":           {"name":"RecoDecayVertex.position.y",  "title":"Reco decay vertex y [mm]",            "bin":100,"xmin":-1000 ,"xmax":1000},
    "Reco_DecayVertex_z":           {"name":"RecoDecayVertex.position.z",  "title":"Reco decay vertex z [mm]",            "bin":100,"xmin":-1000 ,"xmax":1000},
    "Reco_DecayVertex_x_prompt":    {"name":"RecoDecayVertex.position.x",  "title":"Reco decay vertex x [mm]",            "bin":100,"xmin":-0.01 ,"xmax":0.01},
    "Reco_DecayVertex_y_prompt":    {"name":"RecoDecayVertex.position.y",  "title":"Reco decay vertex y [mm]",            "bin":100,"xmin":-0.01 ,"xmax":0.01},
    "Reco_DecayVertex_z_prompt":    {"name":"RecoDecayVertex.position.z",  "title":"Reco decay vertex z [mm]",            "bin":100,"xmin":-0.01 ,"xmax":0.01},
    "Reco_DecayVertex_chi2":        {"name":"RecoDecayVertex.chi2",        "title":"Reco decay vertex #chi^{2}",          "bin":100,"xmin":0 ,"xmax":3},
    "Reco_DecayVertex_probability": {"name":"RecoDecayVertex.probability", "title":"Reco decay vertex probability",       "bin":100,"xmin":0 ,"xmax":10},
    "Reco_Lxy":                     {"name":"Reco_Lxy",                    "title":"Reco L_{xy} [mm]",     "bin":100,"xmin":0 ,"xmax":1000},
    "Reco_Lxyz":                    {"name":"Reco_Lxyz",                   "title":"Reco L_{xyz} [mm]",    "bin":100,"xmin":0 ,"xmax":1000},
    "Reco_Lxyz_prompt":             {"name":"Reco_Lxyz",                   "title":"Reco L_{xyz} [mm]",    "bin":100,"xmin":0 ,"xmax":0.1},

    "Reco_ee_invMass":   {"name":"Reco_ee_invMass",   "title":"Reco m_{ee} [GeV]",           "bin":100,"xmin":0, "xmax":100},

    "RecoPhoton_e":        {"name":"RecoPhoton_e",        "title":"Reco photon energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoPhoton_p":        {"name":"RecoPhoton_p",        "title":"Reco photon p [GeV]",      "bin":100,"xmin":0 ,"xmax":50},
    "RecoPhoton_pt":       {"name":"RecoPhoton_pt",       "title":"Reco photon p_{T} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoPhoton_pz":       {"name":"RecoPhoton_pz",       "title":"Reco photon p_{z} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoPhoton_eta":      {"name":"RecoPhoton_eta",      "title":"Reco photon #eta",         "bin":60, "xmin":-3,"xmax":3},
    "RecoPhoton_theta":    {"name":"RecoPhoton_theta",    "title":"Reco photon #theta",       "bin":64, "xmin":0,"xmax":3.2},
    "RecoPhoton_phi":      {"name":"RecoPhoton_phi",      "title":"Reco photon #phi",         "bin":64, "xmin":-3.2,"xmax":3.2},
    "RecoPhoton_charge":   {"name":"RecoPhoton_charge",   "title":"Reco photon charge",       "bin":3, "xmin":-1.5,"xmax":1.5},

    "RecoMuon_e":        {"name":"RecoMuon_e",        "title":"Reco muon energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoMuon_p":        {"name":"RecoMuon_p",        "title":"Reco muon p [GeV]",      "bin":100,"xmin":0 ,"xmax":50},
    "RecoMuon_pt":       {"name":"RecoMuon_pt",       "title":"Reco muon p_{T} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoMuon_pz":       {"name":"RecoMuon_pz",       "title":"Reco muon p_{z} [GeV]",  "bin":100,"xmin":0 ,"xmax":50},
    "RecoMuon_eta":      {"name":"RecoMuon_eta",      "title":"Reco muon #eta",         "bin":60, "xmin":-3,"xmax":3},
    "RecoMuon_theta":    {"name":"RecoMuon_theta",    "title":"Reco muon #theta",       "bin":64, "xmin":0,"xmax":3.2},
    "RecoMuon_phi":      {"name":"RecoMuon_phi",      "title":"Reco muon #phi",         "bin":64, "xmin":-3.2,"xmax":3.2},
    "RecoMuon_charge":   {"name":"RecoMuon_charge",   "title":"Reco muon charge",       "bin":3, "xmin":-1.5,"xmax":1.5},

    "RecoMuonTrack_absD0":             {"name":"RecoMuonTrack_absD0",     "title":"Reco muon tracks |d_{0}| [mm]",      "bin":100,"xmin":0, "xmax":2000},
    "RecoMuonTrack_absD0_prompt":      {"name":"RecoMuonTrack_absD0",     "title":"Reco muon tracks |d_{0}| [mm]",      "bin":100,"xmin":0, "xmax":1},
    "RecoMuonTrack_absZ0":             {"name":"RecoMuonTrack_absZ0",     "title":"Reco muon tracks |z_{0}| [mm]",      "bin":100,"xmin":0, "xmax":2000},
    "RecoMuonTrack_absZ0_prompt":      {"name":"RecoMuonTrack_absZ0",     "title":"Reco muon tracks |z_{0}| [mm]",      "bin":100,"xmin":0, "xmax":1},
    "RecoMuonTrack_absD0sig":          {"name":"RecoMuonTrack_absD0sig",  "title":"Reco muon tracks |d_{0} significance|",      "bin":100,"xmin":0, "xmax":600000},
    "RecoMuonTrack_absD0sig_prompt":   {"name":"RecoMuonTrack_absD0sig",  "title":"Reco muon tracks |d_{0} significance|",      "bin":100,"xmin":0, "xmax":5},
    "RecoMuonTrack_absZ0sig":          {"name":"RecoMuonTrack_absZ0sig",  "title":"Reco muon tracks |z_{0} significance|",      "bin":100,"xmin":0, "xmax":600000},
    "RecoMuonTrack_absZ0sig_prompt":   {"name":"RecoMuonTrack_absZ0sig",  "title":"Reco muon tracks |z_{0} significance|",      "bin":100,"xmin":0, "xmax":5},
    "RecoMuonTrack_D0cov":      {"name":"RecoMuonTrack_D0cov",     "title":"Reco muon tracks d_{0} #sigma^{2}",      "bin":100,"xmin":0, "xmax":0.5},
    "RecoMuonTrack_Z0cov":      {"name":"RecoMuonTrack_Z0cov",     "title":"Reco muon tracks z_{0} #sigma^{2}",      "bin":100,"xmin":0, "xmax":0.5},

    "RecoMissingEnergy_e":       {"name":"RecoMissingEnergy_e",       "title":"Reco Total Missing Energy [GeV]",    "bin":100,"xmin":0 ,"xmax":50},
    "RecoMissingEnergy_p":       {"name":"RecoMissingEnergy_p",       "title":"Reco Total Missing p [GeV]",         "bin":100,"xmin":0 ,"xmax":50},
    "RecoMissingEnergy_pt":      {"name":"RecoMissingEnergy_pt",      "title":"Reco Missing p_{T} [GeV]",           "bin":100,"xmin":0 ,"xmax":50},
    "RecoMissingEnergy_px":      {"name":"RecoMissingEnergy_px",      "title":"Reco Missing p_{x} [GeV]",           "bin":100,"xmin":0 ,"xmax":50},
    "RecoMissingEnergy_py":      {"name":"RecoMissingEnergy_py",      "title":"Reco Missing p_{y} [GeV]",           "bin":100,"xmin":0 ,"xmax":50},
    "RecoMissingEnergy_pz":      {"name":"RecoMissingEnergy_pz",      "title":"Reco Missing p_{z} [GeV]",           "bin":100,"xmin":0 ,"xmax":50},
    "RecoMissingEnergy_eta":     {"name":"RecoMissingEnergy_eta",     "title":"Reco Missing Energy #eta",           "bin":60,"xmin":-3 ,"xmax":3},
    "RecoMissingEnergy_theta":   {"name":"RecoMissingEnergy_theta",   "title":"Reco Missing Energy #theta",         "bin":64,"xmin":0 , "xmax":3.2},
    "RecoMissingEnergy_phi":     {"name":"RecoMissingEnergy_phi",     "title":"Reco Missing Energy #phi",           "bin":64,"xmin":-3.2 ,"xmax":3.2},

}
