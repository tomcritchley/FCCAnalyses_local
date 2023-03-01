#Mandatory: List of processes
processList = {

        #centrally-produced backgrounds
        #'p8_ee_Zee_ecm91':{'chunks':100},
        #'p8_ee_Zbb_ecm91':{'chunks':100},
        #'p8_ee_Ztautau_ecm91':{'chunks':100},
        #'p8_ee_Zuds_ecm91':{'chunks':100},
        #'p8_ee_Zcc_ecm91':{'chunks':100},

        #privately-produced signals
        #'eenu_30GeV_1p41e-6Ve':{},
        #'eenu_50GeV_1p41e-6Ve':{},
        #'eenu_70GeV_1p41e-6Ve':{},
        #'eenu_90GeV_1p41e-6Ve':{},

        #test
        #'p8_ee_Zee_ecm91':{'fraction':0.000001},
        #'p8_ee_Zuds_ecm91':{'chunks':10,'fraction':0.000001},
        'p8_ee_Zbb_ecm91':{'fraction': 0.00001},
}

#Production tag. This points to the yaml files for getting sample statistics
#Mandatory when running over EDM4Hep centrally produced events
#Comment out when running over privately produced events
prodTag     = "FCCee/winter2023/IDEA/"

#Input directory
#Comment out when running over centrally produced events
#Mandatory when running over privately produced events
#inputDir = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_Zbb_ecm91/"


#Optional: output directory, default is local dir
outputDir = "outputs/HNL_Dirac_ejj_50GeV_1e-3Ve_W2023_test/output_stage1/"
#outputDir = "/eos/user/j/jalimena/FCCeeLLP/"
#outputDir = "output_stage1/"

#Optional: ncpus, default is 4
nCPUS       = 4

#Optional running on HTCondor, default is False
runBatch    = False
#runBatch    = True

#Optional batch queue name when running on HTCondor, default is workday
#batchQueue = "longlunch"

#Optional computing account when running on HTCondor, default is group_u_FCC.local_gen
#compGroup = "group_u_FCC.local_gen"

#Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis():
        def analysers(df):

                df2 = (df

                #Access the various objects and their properties with the following syntax: .Define("<your_variable>", "<accessor_fct (name_object)>")
		#This will create a column in the RDataFrame named <your_variable> and filled with the return value of the <accessor_fct> for the given collection/object 
		#Accessor functions are the functions found in the C++ analyzers code that return a certain variable, e.g. <namespace>::get_n(object) returns the number 
		#of these objects in the event and <namespace>::get_pt(object) returns the pt of the object. Here you can pick between two namespaces to access either
		#reconstructed (namespace = ReconstructedParticle) or MC-level objects (namespace = MCParticle). 
		#For the name of the object, in principle the names of the EDM4HEP collections are used - photons, muons and electrons are an exception, see below

		#OVERVIEW: Accessing different objects and counting them
               

                # Following code is written specifically for the HNL study
                ####################################################################################################
                .Alias("Particle1", "Particle#1.index")
                .Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
                .Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")

                .Define("n_RecoTracks","ReconstructedParticle2Track::getTK_n(EFlowTrack_1)") 
                #all final state gen electrons and positrons
                #all final state gen electrons and positrons
                .Define("GenElectron_PID", "FCCAnalyses::MCParticle::sel_pdgID(11, true)(Particle)")
                .Define("FSGenElectron", "FCCAnalyses::MCParticle::sel_genStatus(1)(GenElectron_PID)") #gen status==1 means final state particle (FS)
                .Define("n_FSGenElectron", "FCCAnalyses::MCParticle::get_n(FSGenElectron)")
                #put in dummy values below if there aren't any FSGenElectrons, to avoid seg fault
                .Define("FSGenElectron_e", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_e(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_p", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_p(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_pt", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_pt(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_px", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_px(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_py", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_py(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_pz", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_pz(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_eta", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_eta(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_theta", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_theta(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_phi", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_phi(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_charge", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_charge(FSGenElectron); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
               
                #Jets
                .Define("n_RecoJets", "ReconstructedParticle::get_n(Jet)")

                .Define("RecoJet_e",      "ReconstructedParticle::get_e(Jet)")
                .Define("RecoJet_pt",      "ReconstructedParticle::get_pt(Jet)")
                .Define("RecoJet_eta",     "ReconstructedParticle::get_eta(Jet)")
                .Define("RecoJet_phi",     "ReconstructedParticle::get_phi(Jet)")
       
                .Define("RecoLeadJet_e",  "if (n_RecoJets >= 1) return (RecoJet_e.at(0)); else return float(-10000.);")
                .Define("RecoLeadJet_pt",  "if (n_RecoJets >= 1) return (RecoJet_pt.at(0)); else return float(-10000.);")
                .Define("RecoLeadJet_phi",  "if (n_RecoJets >= 1) return (RecoJet_phi.at(0)); else return float(-10000.);")
                .Define("RecoLeadJet_eta",  "if (n_RecoJets >= 1) return (RecoJet_eta.at(0)); else return float(-10000.);") 
                .Define("RecoLeadingJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoLeadJet_pt, RecoLeadJet_eta, RecoLeadJet_phi, RecoLeadJet_e)")

                .Define("RecoSecondJet_e",  "if (n_RecoJets > 1) return (RecoJet_e.at(1)); else return float(-1000.);")
                .Define("RecoSecondJet_pt",  "if (n_RecoJets > 1) return (RecoJet_pt.at(1)); else return float(-1000.);")
                .Define("RecoSecondJet_phi",  "if (n_RecoJets > 1) return (RecoJet_phi.at(1)); else return float(1000.);")
                .Define("RecoSecondJet_eta",  "if (n_RecoJets > 1) return (RecoJet_eta.at(1)); else return float(1000.);")
                .Define("RecoSecondJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoSecondJet_pt, RecoSecondJet_eta, RecoSecondJet_phi, RecoSecondJet_e)")
                #Build Di-Jet 4 Vector
                .Define("RecoDiJet4Vect", "ReconstructedParticle::get_tlv_sum(RecoLeadingJet4Vect, RecoSecondJet4Vect)")
                .Define("RecoDiJet_invMass", "ReconstructedParticle::get_tlv_mass(RecoDiJet4Vect)")                

                #Electron
                .Alias("Electron0", "Electron#0.index")
		.Define("RecoElectrons",  "ReconstructedParticle::get(Electron0, ReconstructedParticles)")
		.Define("n_RecoElectrons",  "ReconstructedParticle::get_n(RecoElectrons)") #count how many electrons are in the event in total

                .Define("RecoElectron_e",      "ReconstructedParticle::get_e(RecoElectrons)")
                .Define("RecoElectron_pt",      "ReconstructedParticle::get_pt(RecoElectrons)")
                .Define("RecoElectron_eta",     "ReconstructedParticle::get_eta(RecoElectrons)")
                .Define("RecoElectron_phi",     "ReconstructedParticle::get_phi(RecoElectrons)")

                #Get leading electron (Should be the one from the HNL decay)

                .Define("RecoElectron_lead_e", "if (n_RecoElectrons >=1) return ReconstructedParticle::get_e(RecoElectrons).at(0); else return float(-1.);")
                .Define("RecoElectron_lead_pt", "if (n_RecoElectrons >=1) return ReconstructedParticle::get_pt(RecoElectrons).at(0); else return float(-1.);")
                .Define("RecoElectron_lead_eta", "if (n_RecoElectrons >=1) return ReconstructedParticle::get_eta(RecoElectrons).at(0); else return float(-100.);")
                .Define("RecoElectron_lead_phi", "if (n_RecoElectrons >=1) return ReconstructedParticle::get_phi(RecoElectrons).at(0); else return float(-100.);")
                .Define("RecoElectron4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoElectron_lead_pt, RecoElectron_lead_eta, RecoElectron_lead_phi, RecoElectron_lead_e)")

                #Build DiJet + Electron 4-Vector
                .Define("RecoDiJetElectron4Vect", "ReconstructedParticle::get_tlv_sum(RecoDiJet4Vect, RecoElectron4Vect)")
                .Define("RecoDiJetElectron_invmass", "ReconstructedParticle::get_tlv_mass(RecoDiJetElectron4Vect)")


)
                return df2

        def output():
                branchList = [
                        "FSGenElectron_e",
                        "n_RecoJets",
                        "RecoDiJet_invMass",
                        "n_RecoTracks",
                        "n_RecoElectrons",
                        "RecoDiJetElectron_invmass",
		]

                return branchList
