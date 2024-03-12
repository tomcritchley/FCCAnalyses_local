#Input directory where the files produced at the stage1 level are
inputDir = "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/HNL_sample_creation/signal_HNLS"

outputDir = "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/HNL_sample_creation/signal_HNLS/final_sig"

processList = {
    #run over the full statistics from stage1
    #'ejjnu':{},
    #'p8_ee_Zbb_ecm91':{},
    #'p8_ee_Zcc_ecm91':{},
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

}

###Dictionary for the ouput variable/hitograms. The key is the name of the variable in the output files. "name" is the name of the variable in the input file, "title" is the x-axis label of the histogram, "bin" the number of bins of the histogram, "xmin" the minimum x-axis value and "xmax" the maximum x-axis value.
histoList = {

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

    
    #"RecoDiJet_invMass":   {"name":"RecoDiJet_invMass",   "title":"Reco DiJet mass [GeV]",           "bin":100,"xmin":0, "xmax":100},
    #"RecoDiJetElectron_invMass":   {"name":"RecoDiJetElectron_invMass",   "title":"Reco DiJet-Electrons mass [GeV]",           "bin":100,"xmin":0, "xmax":100},
    #"RecoDiJet_electron_invMass":   {"name":"RecoDiJet_electron_invMass",   "title":"Reco DiJet-electron (-) mass [GeV]",           "bin":100,"xmin":0, "xmax":100},
    #"RecoDiJet_positron_invMass":   {"name":"RecoDiJet_positron_invMass",   "title":"Reco DiJet-positron (-) mass [GeV]",           "bin":100,"xmin":0, "xmax":100},

    "RecoElectron_lead_e":        {"name":"RecoElectron_lead_e",        "title":"Reco Electron (from HNL) energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},

    "RecoElectronTrack_absD0":             {"name":"RecoElectronTrack_absD0",     "title":"Reco positron tracks |d_{0}| [mm]",      "bin":100,"xmin":0, "xmax":700},

    "RecoElectronTrack_absD0sig":          {"name":"RecoElectronTrack_absD0sig",  "title":"Reco positron tracks |d_{0} significance|",      "bin":100,"xmin":0, "xmax":50},
   
    "RecoMissingEnergy_e":       {"name":"RecoMissingEnergy_e",       "title":"Reco Total Missing Energy [GeV]",    "bin":100,"xmin":0 ,"xmax":50},
    "RecoMissingEnergy_theta":   {"name":"RecoMissingEnergy_theta",   "title":"Reco Missing Energy #theta",         "bin":64,"xmin":0 , "xmax":3.2},

    "Vertex_chi2": {"name":"Vertex_chi2",       "title":"Chi2 of the primary vertex",    "bin":100,"xmin":0 ,"xmax":50},
    "n_primt": {"name":"n_primt",       "title":"Number of primary tracks",    "bin":100,"xmin":0 ,"xmax":50},
    "ntracks": {"name":"ntracks",       "title":"Number of total tracks",    "bin":100,"xmin":0 ,"xmax":50},

}
