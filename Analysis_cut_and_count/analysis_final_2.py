#Input directory where the files produced at the stage1 level are

inputDir = "/eos/user/t/tcritchl/HNLs_Feb24"

outputDir = "/eos/user/t/tcritchl/HNls_Feb24/final/"

processList = {
    #'p8_ee_Zcc_ecm91':{},
    #'p8_ee_Zbb_ecm91':{},
    'HNL_Dirac_ejj_20GeV_1e-3Ve':{},
    'HNL_Dirac_ejj_50GeV_1e-3Ve':{},
    'HNL_Dirac_ejj_70GeV_1e-3Ve':{},
}


#Link to the dictonary that contains all the cross section information etc...
procDict = "FCCee_procDict_winter2023_IDEA.json"

#Add MySample_p8_ee_ZH_ecm240 as it is not an offical process
#procDictAdd={
    #"MySample_p8_ee_ZH_ecm240":{"numberOfEvents": 10000000, "sumOfWeights": 10000000, "crossSection": 0.201868, "kfactor": 1.0, "matchingEfficiency": 1.0}
#    "ejjnu":{"numberOfEvents": 100000, "sumOfWeights": 100000, "crossSection": 0.014, "kfactor": 1.0, "matchingEfficiency": 1.0}
#}

#produces ROOT TTrees, default is False
#doTree = True

##Dictionnay of the list of cuts. The key is the name of the selection that will be added to the output file
cutList = {
    "selNone": "n_RecoTracks > -1",
    "selRecoEleGt0":"n_RecoElectrons > 0", 
    #"selRecoEleEGt35": "RecoElectron_lead_e > 35", 
    "selRecoEleEGt11": "n_RecoElectrons > 0 && RecoElectron_lead_e > 11",
    "selMissingEGt25": "RecoMissingEnergy_e[0] > 25",
    #"selMissingEGt12": "RecoMissingEnergy_e[0] > 12",
    "selEleDiJetDRLt27": "RecoElectron_DiJet_delta_R < 2.7", #Dim thesis
    #"selEleDiJetDRLt3": "RecoElectron_DiJet_delta_R < 3.0", 
    "selEleSecondJetDRGt2": "RecoElectron_SecondJet_delta_R > 2",
    "selEleEGt11_MissingEGt25": "RecoElectron_lead_e > 11 && RecoMissingEnergy_e[0] > 25",
    
    "selEleEGt11_MissingEGt25_EleDiJetDRLt27": "RecoElectron_lead_e > 11 && RecoMissingEnergy_e[0] > 25 && RecoElectron_DiJet_delta_R < 2.7",
    "selEleEGt11_MissingEGt25_EleDiJetDRLt27_EleSecondJetDRGt2": "RecoElectron_lead_e > 11 && RecoMissingEnergy_e[0] > 25 && RecoElectron_DiJet_delta_R < 2.7 && RecoElectron_SecondJet_delta_R > 2",
    "allcuts": "RecoElectron_lead_e > 35 && RecoMissingEnergy_e[0] > 12 && RecoElectron_SecondJet_delta_R > 3",
 	
   #####cuts from dimitri's thesis to plot
   "selMissingEGt12": "RecoMissingEnergy_e[0] > 12",
   "selRecoEleEGt35": "RecoElectron_lead_e > 35",
   "selAngleLt24": "RecoDiJet_angle < 2.4",
   "selEleDiJetDRLt3": "RecoElectron_DiJet_delta_R < 3.0",
   
   #combined cuts from dimitri
   "selMissingEGt12_EleEGt35": "RecoMissingEnergy_e[0] > 12 && RecoElectron_lead_e > 35",
   "selMissingEGt12_EleEGt35_AngleLt24": "RecoMissingEnergy_e[0] > 12 && RecoElectron_lead_e > 35 && RecoDiJet_angle < 2.4",
   "selMissingEGt12_EleEGt35_AngleLt24_DiJetDRLt3": "RecoMissingEnergy_e[0] > 12 && RecoElectron_lead_e > 35 && RecoDiJet_angle < 2.4 && RecoElectron_DiJet_delta_R < 3.0", 
   "selChi2": "Vertex_chi2 < 10",
   "selTracks": "ntracks - nprimt < 5"
}

###Dictionary for the ouput variable/hitograms. The key is the name of the variable in the output files. "name" is the name of the variable in the input file, "title" is the x-axis label of the histogram, "bin" the number of bins of the histogram, "xmin" the minimum x-axis value and "xmax" the maximum x-axis value.
histoList = {

    "RecoDiJet_phi":        {"name":"RecoDiJet_phi",        "title":"Reco di-jet phi]", "bin":64,"xmin":-3.2 ,"xmax":3.2},

    "RecoDiJet_angle":      {"name":"RecoDiJet_angle",      "title":"Reco di-jet angle", "bin":100,"xmin":0 ,"xmax":3.2},
    
    "RecoElectron_DiJet_delta_R":      {"name":"RecoElectron_DiJet_delta_R",     "title":"RecoElectron - DiJet #Delta R",      "bin":100,"xmin":0, "xmax":4.5},
    "RecoDiJet_delta_R":      {"name":"RecoDiJet_delta_R",     "title":"RecoDiJet #Delta R",      "bin":100,"xmin":0, "xmax":4.5},


    "RecoElectron_lead_e":        {"name":"RecoElectron_lead_e",        "title":"Reco Electron (from HNL) energy [GeV]", "bin":100,"xmin":0 ,"xmax":50},
    "RecoElectronTrack_absD0":             {"name":"RecoElectronTrack_absD0",     "title":"Reco positron tracks |d_{0}| [mm]",      "bin":100,"xmin":0, "xmax":700},
    "RecoElectronTrack_absD0sig":          {"name":"RecoElectronTrack_absD0sig",  "title":"Reco positron tracks |d_{0} significance|",      "bin":100,"xmin":0, "xmax":50},

    "RecoMissingEnergy_e":       {"name":"RecoMissingEnergy_e",       "title":"Reco Total Missing Energy [GeV]",    "bin":100,"xmin":0 ,"xmax":50},
    "RecoMissingEnergy_phi":     {"name":"RecoMissingEnergy_phi",     "title":"Reco Missing Energy #phi",           "bin":64,"xmin":-3.2 ,"xmax":3.2},

    "Vertex_chi2":        {"name":"Vertex_chi2",        "title":"vertex #chi^{2}",          "bin":100,"xmin":0 ,"xmax":3},
    "n_primt":        {"name":"RecoDecayVertex.chi2",        "title":"Number of primary tracks",          "bin":5,"xmin":-0.5 ,"xmax":4.5},
    "ntracks":        {"name":"RecoDecayVertex.chi2",        "title":"Number of tracks",          "bin":5,"xmin":-0.5 ,"xmax":4.5},
    
    }
