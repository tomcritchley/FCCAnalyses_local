#Mandatory: List of processes
processList = {

        #centrally-produced backgrounds
        #'p8_ee_Zee_ecm91':{'chunks':100},
        #'p8_ee_Zcc_ecm91':{'chunks':100},
        #'p8_ee_Ztautau_ecm91':{'chunks':100},
        #'p8_ee_Zuds_ecm91':{'chunks':100},
        #'p8_ee_Zcc_ecm91':{'chunks':100},

        #privately-produced signals
        #'eenu_30GeV_1p41e-6Ve':{},
        #'eenu_50GeV_1p41e-6Ve':{},
        #'eenu_70GeV_1p41e-6Ve':{},
        #'eenu_90GeV_1p41e-6Ve':{},

        #test
        #'p8_ee_Zcc_ecm91':{'fraction':0.1},
        #'p8_ee_Zcc_ecm91':{'chunks':100},
        #'p8_ee_Zuds_ecm91':{'chunks':10,'fraction':0.000001},
        #'p8_ee_Zud_ecm91':{'chunks':10,'fraction':0.000001},
	#'p8_ee_Zee_ecm91':{},
	#'p8_ee_Zcc_ecm91':{},
	#'p8_ee_Ztautau_ecm91':{}, 
	#'p8_ee_Zud_ecm91':{'fraction':0.000001},
	#'p8_ee_Zcc_ecm91':{'fraction':0.000001},
	'p8_ee_Zbb_ecm91':{'fraction':0.000001},	
}



#Production tag. This points to the yaml files for getting sample statistics
#Mandatory when running over EDM4Hep centrally produced events
#Comment out when running over privately produced events
prodTag     = "FCCee/winter2023/IDEA/"

#Input directory
#Comment out when running over centrally produced events
#Mandatory when running over privately produced events
#inputDir = "/eos/experiment/fcc/ee/analyses/case-studies/bsm/LLPs/HNLs/HNL_eenu_MadgraphPythiaDelphes"


#Optional: output directory, default is local dir
#outputDir = "outputs/p8_ee_Zcc_ecm91_W2023/output_stage1/"
#outputDir = "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/Analysis"
#outputDir = "./outputs/"
#outputDir= "/eos/user/t/tcritchl/output_stage1_panTest"

HNL_id = "9990012" # Dirac
#HNL_id = "9900012" # Majorana

#Optional: ncpus, default is 4
#nCPUS       = 4

#Optional running on HTCondor, default is False
runBatch    = False
#runBatch    = True

#Optional batch queue name when running on HTCondor, default is workday
#batchQueue = "longlunch"

#Optional computing account when running on HTCondor, default is group_u_FCC.local_gen
#compGroup = "group_u_FCC.local_gen"
#compGroup = "group_u_BE.ABP.SLAP"
#compGroup = "group_u_ATLAS.u_zp"


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

                .Define("FSGenElectron_vertex_x", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_vertex_x( FSGenElectron ); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_vertex_y", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_vertex_y( FSGenElectron ); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")
                .Define("FSGenElectron_vertex_z", "if (n_FSGenElectron>0) return FCCAnalyses::MCParticle::get_vertex_z( FSGenElectron ); else return FCCAnalyses::MCParticle::get_genStatus(GenElectron_PID);")

                # Finding the Lxy of the HNL
                # Definition: Lxy = math.sqrt( (branchGenPtcl.At(daut1).X)**2 + (branchGenPtcl.At(daut1).Y)**2 )
                .Define("FSGen_Lxy", "return sqrt(FSGenElectron_vertex_x*FSGenElectron_vertex_x + FSGenElectron_vertex_y*FSGenElectron_vertex_y)")
                # Finding the Lxyz of the HNL
                .Define("FSGen_Lxyz", "return sqrt(FSGenElectron_vertex_x*FSGenElectron_vertex_x + FSGenElectron_vertex_y*FSGenElectron_vertex_y + FSGenElectron_vertex_z*FSGenElectron_vertex_z)")

                #all final state gen neutrinos and anti-neutrinos
                .Define("GenNeutrino_PID", "FCCAnalyses::MCParticle::sel_pdgID(12, true)(Particle)")
                .Define("FSGenNeutrino", "FCCAnalyses::MCParticle::sel_genStatus(1)(GenNeutrino_PID)") #gen status==1 means final state particle (FS)
                .Define("n_FSGenNeutrino", "FCCAnalyses::MCParticle::get_n(FSGenNeutrino)")
                .Define("FSGenNeutrino_e", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_e(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_p", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_p(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_pt", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_pt(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_px", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_px(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_py", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_py(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_pz", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_pz(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_eta", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_eta(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_theta", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_theta(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_phi", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_phi(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")
                .Define("FSGenNeutrino_charge", "if (n_FSGenNeutrino>0) return FCCAnalyses::MCParticle::get_charge(FSGenNeutrino); else return FCCAnalyses::MCParticle::get_genStatus(GenNeutrino_PID);")

                #all final state gen photons
                .Define("GenPhoton_PID", "FCCAnalyses::MCParticle::sel_pdgID(22, false)(Particle)")
                .Define("FSGenPhoton", "FCCAnalyses::MCParticle::sel_genStatus(1)(GenPhoton_PID)") #gen status==1 means final state particle (FS)
                .Define("n_FSGenPhoton", "FCCAnalyses::MCParticle::get_n(FSGenPhoton)")
                .Define("FSGenPhoton_e", "FCCAnalyses::MCParticle::get_e(FSGenPhoton)")
                .Define("FSGenPhoton_p", "FCCAnalyses::MCParticle::get_p(FSGenPhoton)")
                .Define("FSGenPhoton_pt", "FCCAnalyses::MCParticle::get_pt(FSGenPhoton)")
                .Define("FSGenPhoton_px", "FCCAnalyses::MCParticle::get_px(FSGenPhoton)")
                .Define("FSGenPhoton_py", "FCCAnalyses::MCParticle::get_py(FSGenPhoton)")
                .Define("FSGenPhoton_pz", "FCCAnalyses::MCParticle::get_pz(FSGenPhoton)")
                .Define("FSGenPhoton_eta", "FCCAnalyses::MCParticle::get_eta(FSGenPhoton)")
                .Define("FSGenPhoton_theta", "FCCAnalyses::MCParticle::get_theta(FSGenPhoton)")
                .Define("FSGenPhoton_phi", "FCCAnalyses::MCParticle::get_phi(FSGenPhoton)")
                .Define("FSGenPhoton_charge", "FCCAnalyses::MCParticle::get_charge(FSGenPhoton)")


                # ee invariant mass
                .Define("FSGen_ee_energy", "if (n_FSGenElectron>1) return (FSGenElectron_e.at(0) + FSGenElectron_e.at(1)); else return float(-1.);")
                .Define("FSGen_ee_px", "if (n_FSGenElectron>1) return (FSGenElectron_px.at(0) + FSGenElectron_px.at(1)); else return float(-1.);")
                .Define("FSGen_ee_py", "if (n_FSGenElectron>1) return (FSGenElectron_py.at(0) + FSGenElectron_py.at(1)); else return float(-1.);")
                .Define("FSGen_ee_pz", "if (n_FSGenElectron>1) return (FSGenElectron_pz.at(0) + FSGenElectron_pz.at(1)); else return float(-1.);")
                .Define("FSGen_ee_invMass", "if (n_FSGenElectron>1) return sqrt(FSGen_ee_energy*FSGen_ee_energy - FSGen_ee_px*FSGen_ee_px - FSGen_ee_py*FSGen_ee_py - FSGen_ee_pz*FSGen_ee_pz ); else return float(-1.);")

                # eenu invariant mass
                .Define("FSGen_eenu_energy", "if (n_FSGenElectron>1 && n_FSGenNeutrino>0) return (FSGenElectron_e.at(0) + FSGenElectron_e.at(1) + FSGenNeutrino_e.at(0)); else return float(-1.);")
                .Define("FSGen_eenu_px", "if (n_FSGenElectron>1 && n_FSGenNeutrino>0) return (FSGenElectron_px.at(0) + FSGenElectron_px.at(1) + FSGenNeutrino_px.at(0)); else return float(-1.);")
                .Define("FSGen_eenu_py", "if (n_FSGenElectron>1 && n_FSGenNeutrino>0) return (FSGenElectron_py.at(0) + FSGenElectron_py.at(1) + FSGenNeutrino_py.at(0)); else return float(-1.);")
                .Define("FSGen_eenu_pz", "if (n_FSGenElectron>1 && n_FSGenNeutrino>0) return (FSGenElectron_pz.at(0) + FSGenElectron_pz.at(1) + FSGenNeutrino_pz.at(0)); else return float(-1.);")
                .Define("FSGen_eenu_invMass", "if (n_FSGenElectron>1 && n_FSGenNeutrino>0) return sqrt(FSGen_eenu_energy*FSGen_eenu_energy - FSGen_eenu_px*FSGen_eenu_px - FSGen_eenu_py*FSGen_eenu_py - FSGen_eenu_pz*FSGen_eenu_pz ); else return float(-1.);")
                 
                 #ELECTRONS AND MUONS
                .Alias("Electron0", "Electron#0.index")
                .Define("RecoElectrons",  "ReconstructedParticle::get(Electron0, ReconstructedParticles)")
                .Define("n_RecoElectrons",  "ReconstructedParticle::get_n(RecoElectrons)") #count how many electrons are in the event in total

                .Alias("Muon0", "Muon#0.index")
                .Define("RecoMuons",  "ReconstructedParticle::get(Muon0, ReconstructedParticles)")
                .Define("n_RecoMuons",  "ReconstructedParticle::get_n(RecoMuons)") #count how many muons are in the event in total

                # MC event primary vertex
                .Define("MC_PrimaryVertex",  "FCCAnalyses::MCParticle::get_EventPrimaryVertex(21)( Particle )" )

                # Reconstructed particles
                .Define("n_RecoTracks","ReconstructedParticle2Track::getTK_n(EFlowTrack_1)")
                
		.Define("GenMuon_PID", "FCCAnalyses::MCParticle::sel_pdgID(13, true)(Particle)")
                .Define("FSGenMuon", "FCCAnalyses::MCParticle::sel_genStatus(1)(GenMuon_PID)") #gen status==1 means final state particle (FS)


                #JETS
                #.Define("n_GenJets" , "ReconstructedParticle::get_n(GenJet)") # Count number of jets per event (gen level)

                #Define Gen-level jets
                #Remove electrons/muons from jets
                .Define("MCParticles", "FCCAnalyses::MCParticle::sel_genStatus(1)(Particle)") #select final state particles
                .Define("n_MCParticles", "FCCAnalyses::MCParticle::get_n(MCParticles)")

                .Define("Jet_GenParticles0", "MCParticle::remove(MCParticles, FSGenElectron)") #remove electron
                .Define("Jet_GenParticles", "MCParticle::remove(Jet_GenParticles0, FSGenNeutrino)") #remove neutrino

                .Define("GP_px",          "FCCAnalyses::MCParticle::get_px(Jet_GenParticles)")
                .Define("GP_py",          "FCCAnalyses::MCParticle::get_py(Jet_GenParticles)")
                .Define("GP_pz",          "FCCAnalyses::MCParticle::get_pz(Jet_GenParticles)")
                .Define("GP_e",           "FCCAnalyses::MCParticle::get_e(Jet_GenParticles)")

                .Define("GenPseudo_jets",    "JetClusteringUtils::set_pseudoJets(GP_px, GP_py, GP_pz, GP_e)")

                .Define("FCCAnalysesGenJets_eekt", "JetClustering::clustering_ee_kt(2, 2, 0, 0)(GenPseudo_jets)")
                .Define("GenJets_ee_kt",           "JetClusteringUtils::get_pseudoJets(FCCAnalysesGenJets_eekt)")

                .Define("GenJets_ee_kt_n", "JetClusteringUtils::get_n(GenJets_ee_kt)")
                .Define("GenJets_ee_kt_e", "JetClusteringUtils::get_e(GenJets_ee_kt)")
                .Define("GenJets_ee_kt_pt",        "JetClusteringUtils::get_pt(GenJets_ee_kt)")
                .Define("GenJets_ee_kt_eta",        "JetClusteringUtils::get_eta(GenJets_ee_kt)")
                .Define("GenJets_ee_kt_phi",        "JetClusteringUtils::get_phi(GenJets_ee_kt)")
                .Define("GenJets_ee_kt_theta",        "JetClusteringUtils::get_theta(GenJets_ee_kt)")

                .Define("GenLeadJet_e", "if (GenJets_ee_kt_n > 0) return GenJets_ee_kt_e.at(0); else return float(-1.)")
                .Define("GenLeadJet_pt", "if (GenJets_ee_kt_n > 0) return GenJets_ee_kt_pt.at(0); else return float(-1.)")
                .Define("GenLeadJet_eta", "if (GenJets_ee_kt_n > 0) return GenJets_ee_kt_eta.at(0); else return float(-100.)")
                .Define("GenLeadJet_phi", "if (GenJets_ee_kt_n > 0) return GenJets_ee_kt_phi.at(0); else return float(-100.)")
                .Define("GenLeadJet_theta", "if (GenJets_ee_kt_n > 0) return GenJets_ee_kt_theta.at(0); else return float(-100.)")

                .Define("GenSecondJet_e", "if (GenJets_ee_kt_n > 1) return GenJets_ee_kt_e.at(1); else return float(-1.)")
                .Define("GenSecondJet_pt", "if (GenJets_ee_kt_n > 1) return GenJets_ee_kt_pt.at(1); else return float(-1.)")
                .Define("GenSecondJet_eta", "if (GenJets_ee_kt_n > 1) return GenJets_ee_kt_eta.at(1); else return float(-100.)")
                .Define("GenSecondJet_phi", "if (GenJets_ee_kt_n > 1) return GenJets_ee_kt_phi.at(1); else return float(-100.)")
                .Define("GenSecondJet_theta", "if (GenJets_ee_kt_n > 1) return GenJets_ee_kt_theta.at(1); else return float(-100.)")

                #Remove electrons from jets :
                .Define("Jet_ReconstructedParticles", "ReconstructedParticle::remove(ReconstructedParticles, RecoElectrons)")                  

                .Define("RP_px",          "ReconstructedParticle::get_px(Jet_ReconstructedParticles)")
                .Define("RP_py",          "ReconstructedParticle::get_py(Jet_ReconstructedParticles)")
                .Define("RP_pz",          "ReconstructedParticle::get_pz(Jet_ReconstructedParticles)")
                .Define("RP_e",           "ReconstructedParticle::get_e(Jet_ReconstructedParticles)")

                .Define("RecoPseudo_jets",    "JetClusteringUtils::set_pseudoJets(RP_px, RP_py, RP_pz, RP_e)")

                #.Define("FCCAnalysesRecoJets_eekt", "JetClustering::clustering_antikt(1, 2, 2, 0, 0)(RecoPseudo_jets)")
                .Define("FCCAnalysesRecoJets_eekt", "JetClustering::clustering_ee_kt(2, 2, 0, 0)(RecoPseudo_jets)")
                .Define("RecoJets_ee_kt",           "JetClusteringUtils::get_pseudoJets(FCCAnalysesRecoJets_eekt)")

                .Define("RecoJets_ee_kt_n", "JetClusteringUtils::get_n(RecoJets_ee_kt)")
                .Define("RecoJets_ee_kt_e",        "JetClusteringUtils::get_e(RecoJets_ee_kt)")
                .Define("RecoJets_ee_kt_pt",        "JetClusteringUtils::get_pt(RecoJets_ee_kt)")
                .Define("RecoJets_ee_kt_eta",        "JetClusteringUtils::get_eta(RecoJets_ee_kt)")
                .Define("RecoJets_ee_kt_phi",        "JetClusteringUtils::get_phi(RecoJets_ee_kt)")
                .Define("RecoJets_ee_kt_theta",        "JetClusteringUtils::get_theta(RecoJets_ee_kt)")

                .Define("RecoLeadJet_e", "if (RecoJets_ee_kt_n > 0) return RecoJets_ee_kt_e.at(0); else return float(-1.)")
                .Define("RecoLeadJet_pt", "if (RecoJets_ee_kt_n > 0) return RecoJets_ee_kt_pt.at(0); else return float(-1.)")
                .Define("RecoLeadJet_eta", "if (RecoJets_ee_kt_n > 0) return RecoJets_ee_kt_eta.at(0); else return float(-100.)")
                .Define("RecoLeadJet_phi", "if (RecoJets_ee_kt_n > 0) return RecoJets_ee_kt_phi.at(0); else return float(-100.)")
                .Define("RecoLeadJet_theta", "if (RecoJets_ee_kt_n > 0) return RecoJets_ee_kt_theta.at(0); else return float(-100.)")

                .Define("RecoSecondJet_e", "if (RecoJets_ee_kt_n > 1) return RecoJets_ee_kt_e.at(1); else return float(-1.)")
                .Define("RecoSecondJet_pt", "if (RecoJets_ee_kt_n > 1) return RecoJets_ee_kt_pt.at(1); else return float(-1.)")
                .Define("RecoSecondJet_eta", "if (RecoJets_ee_kt_n > 1) return RecoJets_ee_kt_eta.at(1); else return float(-100.)")
                .Define("RecoSecondJet_phi", "if (RecoJets_ee_kt_n > 1) return RecoJets_ee_kt_phi.at(1); else return float(-100.)")
                .Define("RecoSecondJet_theta", "if (RecoJets_ee_kt_n > 1) return RecoJets_ee_kt_theta.at(1); else return float(-100.)")
	

            	#PHOTONS
		.Alias("Photon0", "Photon#0.index") 
		.Define("RecoPhotons",  "ReconstructedParticle::get(Photon0, ReconstructedParticles)")
		.Define("n_RecoPhotons",  "ReconstructedParticle::get_n(RecoPhotons)") #count how many photons are in the event in total

                #.Define("Reco_selected_Jets", "ReconstructedParticle::sel_pt(20.)(Jet)") #select only jets with a pt > 20 GeV 
                #.Define("Reco_selectedJet_e",      "ReconstructedParticle::get_e(Reco_selected_Jets)")
                #.Define("Reco_selectedJet_pt",      "ReconstructedParticle::get_pt(Reco_selected_Jets)")               
                #.Define("Reco_selectedJet_n", "ReconstructedParticle::get_n(Reco_selected_Jets)")
 
                # Difference between lead and secondary jet
                #.Define("GenJetDelta_e", "return (GenLeadJet_e - GenSecondJet_e)")
                #.Define("GenJetDelta_pt", "return (GenLeadJet_pt - GenSecondJet_pt)")
                #.Define("GenJetDelta_phi", "if (GenLeadJet_phi > -1000) return atan2(sin(GenLeadJet_phi - GenSecondJet_phi), cos(GenLeadJet_phi - GenSecondJet_phi)); else return float(100.);")
                #.Define("GenJetDelta_eta", "return (GenLeadJet_eta - GenSecondJet_eta)")
                #.Define("GenJetDelta_R", "return sqrt(GenJetDelta_phi*GenJetDelta_phi + GenJetDelta_eta*GenJetDelta_eta)")
 

                # Building gen-level di-jet 4-vect
                .Define("GenLeadJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenLeadJet_pt, GenLeadJet_eta, GenLeadJet_phi, GenLeadJet_e)")
                .Define("GenSecondJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenSecondJet_pt, GenSecondJet_eta, GenSecondJet_phi, GenSecondJet_e)")
                .Define("GenDiJet", "ReconstructedParticle::get_tlv_sum(GenLeadJet4Vect, GenSecondJet4Vect)")

                # Define gen-level di-jet variables
                .Define("GenDiJet_e", "return ReconstructedParticle::get_tlv_e(GenDiJet).at(0)")
                .Define("GenDiJet_pt", "if (GenDiJet_e > -1) return ReconstructedParticle::get_tlv_pt(GenDiJet).at(0); else return float(-1.);")
                .Define("GenDiJet_eta", "if (GenDiJet_e > -1) return ReconstructedParticle::get_tlv_eta(GenDiJet).at(0); else return float(-100.);")
                .Define("GenDiJet_phi", "if (GenDiJet_e > -1) return ReconstructedParticle::get_tlv_phi(GenDiJet).at(0); else return float(-100.);")
                .Define("GenDiJet_theta", "if (GenDiJet_e > -1) return ReconstructedParticle::get_tlv_theta(GenDiJet).at(0); else return float(-100.);")
                .Define("GenDiJet_px", "if (GenDiJet_e > -1) return ReconstructedParticle::get_tlv_px(GenDiJet).at(0); else return float(-1.);")
                .Define("GenDiJet_py", "if (GenDiJet_e > -1) return ReconstructedParticle::get_tlv_py(GenDiJet).at(0); else return float(-1.);")
                .Define("GenDiJet_pz", "if (GenDiJet_e > -1) return ReconstructedParticle::get_tlv_pz(GenDiJet).at(0); else return float(-1.);")
                .Define("GenDiJet_invMass", "if (GenDiJet_e > -1) return  sqrt(GenDiJet_e*GenDiJet_e - GenDiJet_px*GenDiJet_px - GenDiJet_py*GenDiJet_py - GenDiJet_pz*GenDiJet_pz); else return float(-1.);")
                # Phi analysis study
                #.Define("GenLeadJet_phi_e", "if (GenLeadJet_phi > -1 && GenLeadJet_phi < 1) return float(-1.); else return GenJets_ee_kte.at(0);")
                #.Define("GenLeadJet_phi_pt", "if (GenLeadJet_phi > -1 && GenLeadJet_phi < 1) return float(-1.); else return GenJets_ee_ktpt.at(0);")
                
                #.Define("GenSecondJet_phi_e", "if (GenSecondJet_phi > -1 && GenSecondJet_phi < 1) return float(-1.); else return GenSecondJet_e;")
                #.Define("GenSecondJet_phi_pt", "if (GenSecondJet_phi > -1 && GenSecondJet_phi < 1) return float(-1.); else return GenSecondJet_pt;")

                # Difference between lead and secondary jet
                .Define("RecoJets_ee_ktDelta_e", "return (RecoLeadJet_e - RecoSecondJet_e)")
                .Define("RecoJets_ee_ktDelta_pt", "return (RecoLeadJet_pt - RecoSecondJet_pt)")
                .Define("RecoJets_ee_ktDelta_phi", "if (RecoLeadJet_phi > -10) return atan2(sin(RecoLeadJet_phi - RecoSecondJet_phi), cos(RecoLeadJet_phi - RecoSecondJet_phi)); else return float(100.);")
                .Define("RecoJets_ee_ktDelta_eta", "return (RecoLeadJet_eta - RecoSecondJet_eta)")
                .Define("RecoJets_ee_ktDelta_R", "return sqrt(RecoJets_ee_ktDelta_phi*RecoJets_ee_ktDelta_phi + RecoJets_ee_ktDelta_eta*RecoJets_ee_ktDelta_eta)")

                # Defining vector containing the HNL and its daughters, in order written
                #.Define("GenHNL_indices", "FCCAnalyses::MCParticle::get_indices(%s ,{11} , true, true, true, true)(Particle, Particle1)"%(HNL_id))
                #.Define("GenHNL", "FCCAnalyses::MCParticle::selMC_leg(0)(GenHNL_indices, Particle)")
                #.Define("GenHNL_theta", "FCCAnalyses::MCParticle::get_theta(GenHNL)")

                #.Define("GenHNLElectron", "FCCAnalyses::MCParticle::selMC_leg(1)(GenHNL_indices, Particle)")
                
                # Define electron (e-) 
                .Define("Gen_electron_PID", "FCCAnalyses::MCParticle::sel_pdgID(-11, false)(Particle)")
                .Define("FSGen_electron", "FCCAnalyses::MCParticle::sel_genStatus(1)(Gen_electron_PID)") #gen status==1 means final state particle (FS)
                .Define("FSGen_electron_n", "FCCAnalyses::MCParticle::get_n(FSGen_electron)")
                
                # Define positron (e-)
                .Define("Gen_positron_PID", "FCCAnalyses::MCParticle::sel_pdgID(11, false)(Particle)")
                .Define("FSGen_positron", "FCCAnalyses::MCParticle::sel_genStatus(1)(Gen_positron_PID)") #gen status==1 means final state particle (FS)
                .Define("FSGen_positron_n", "FCCAnalyses::MCParticle::get_n(FSGen_positron)")

                #Define neutrino
                .Define("GenHNL_Neutrino_e", "FCCAnalyses::MCParticle::get_e(FSGenNeutrino)")
                .Define("GenHNL_Neutrino_pt", "FCCAnalyses::MCParticle::get_pt(FSGenNeutrino)")
                .Define("GenHNL_Neutrino_eta", "FCCAnalyses::MCParticle::get_eta(FSGenNeutrino)")
                .Define("GenHNL_Neutrino_phi", "FCCAnalyses::MCParticle::get_phi(FSGenNeutrino)")
 
                .Define("GenNeutrino4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenHNL_Neutrino_pt.at(0), GenHNL_Neutrino_eta.at(0), GenHNL_Neutrino_phi.at(0), GenHNL_Neutrino_e.at(0))")
                .Define("GenNeutrino4Vect_e", "ReconstructedParticle::get_tlv_e(GenNeutrino4Vect)")
          
                #Define decay Electron variables
                #.Define("GenHNLElectron_e", "return FCCAnalyses::MCParticle::get_e(GenHNLElectron).at(0)")
                #.Define("GenHNLElectron_pt", "FCCAnalyses::MCParticle::get_pt(GenHNLElectron).at(0)")
                #.Define("GenHNLElectron_eta", "FCCAnalyses::MCParticle::get_eta(GenHNLElectron).at(0)")
                #.Define("GenHNLElectron_phi", "FCCAnalyses::MCParticle::get_phi(GenHNLElectron).at(0)")
                #.Define("GenHNLElectron_theta", "FCCAnalyses::MCParticle::get_theta(GenHNLElectron)")
                #.Define("GenHNLElectron_charge", "FCCAnalyses::MCParticle::get_charge(GenHNLElectron)")
                #.Define("GenHNLElectron_vertex_x","return FCCAnalyses::MCParticle::get_vertex_x(GenHNLElectron)")
                #.Define("GenHNLElectron_vertex_y","return FCCAnalyses::MCParticle::get_vertex_y(GenHNLElectron)")
                #.Define("GenHNLElectron_vertex_z","return FCCAnalyses::MCParticle::get_vertex_z(GenHNLElectron)")
 
                #Define electron
                #.Define("GenHNL_electron_e", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_e; else return float(-1.);")
                #.Define("GenHNL_electron_pt", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_pt; else return float(-1.);")
                #.Define("GenHNL_electron_phi", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_phi; else return float(-100.);")
                #.Define("GenHNL_electron_eta", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_eta; else return float(-100.);")
                #.Define("GenHNL_electron_theta", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_theta.at(0); else return float(-100.);")

                #Define positron
                #.Define("GenHNL_positron_e", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_e; else return float(-1.);")
                #.Define("GenHNL_positron_pt", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_pt; else return float(-1.);")
                #.Define("GenHNL_positron_phi", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_phi; else return float(-100.);")
                #.Define("GenHNL_positron_eta", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_eta; else return float(-100.);")
                #.Define("GenHNL_positron_theta", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_theta.at(0); else return float(-100.);")

                # Define Di-jet - Electron (e+ and e-) 4 Vect
                #.Define("GenElectron4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenHNLElectron_pt, GenHNLElectron_eta, GenHNLElectron_phi, GenHNLElectron_e)")
                #.Define("GenElectron4Vect_invMass", "ReconstructedParticle::get_tlv_mass(GenElectron4Vect)")
                #.Define("GenDiJetElectron4Vect", "ReconstructedParticle::get_tlv_sum(GenElectron4Vect, GenDiJet)")
                #.Define("GenDiJetElectron_e", "return ReconstructedParticle::get_tlv_e(GenDiJetElectron4Vect).at(0)")
                #.Define("GenDiJetElectron_pt", "if (GenDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_pt(GenDiJetElectron4Vect).at(0); else return float(-1000.);")
                #.Define("GenDiJetElectron_eta", "if (GenDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_eta(GenDiJetElectron4Vect).at(0); else return float(1000.);")
                #.Define("GenDiJetElectron_phi", "if (GenDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_phi(GenDiJetElectron4Vect).at(0); else return float(1000.);")
                #.Define("GenDiJetElectron_theta", "if (GenDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_theta(GenDiJetElectron4Vect).at(0); else return float(1000.);")
                #.Define("GenDiJetElectron_px", "if (GenDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_px(GenDiJetElectron4Vect).at(0); else return float(-1000.);")
                #.Define("GenDiJetElectron_py", "if (GenDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_py(GenDiJetElectron4Vect).at(0); else return float(-1000.);")
                #.Define("GenDiJetElectron_pz", "if (GenDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_pz(GenDiJetElectron4Vect).at(0); else return float(-1000.);")
                #.Define("GenDiJetElectron_invMass", "if (GenDiJetElectron_e > -1) return  sqrt(GenDiJetElectron_e*GenDiJetElectron_e - GenDiJetElectron_px*GenDiJetElectron_px - GenDiJetElectron_py*GenDiJetElectron_py - GenDiJetElectron_pz*GenDiJetElectron_pz); else return float(-1.);")
                #.Define("GenDiJetElectron_invMass_comp", "if (GenDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_mass(GenDiJetElectron4Vect).at(0); else return float(-1.);")

                 # Define Di-jet - electron (e-) 4 Vect
                #.Define("Gen_electron4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenHNL_electron_pt, GenHNL_electron_eta, GenHNL_electron_phi, GenHNL_electron_e)")
                #.Define("GenDiJet_electron4Vect", "ReconstructedParticle::get_tlv_sum(Gen_electron4Vect, GenDiJet)")
                #.Define("GenDiJet_electron_e", "ReconstructedParticle::get_tlv_e(GenDiJet_electron4Vect)")
                #.Define("GenDiJet_electron_theta", "if (GenDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(GenDiJet_electron4Vect).at(0); else return float(1000.);")
                #.Define("GenDiJet_electron_px", "if (GenDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(GenDiJet_electron4Vect).at(0); else return float(-2000.);")
                #.Define("GenDiJet_electron_py", "if (GenDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(GenDiJet_electron4Vect).at(0); else return float(-1000.);")
                #.Define("GenDiJet_electron_pz", "if (GenDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(GenDiJet_electron4Vect).at(0); else return float(-1000.);")
                #.Define("GenDiJet_electron_invMass", "if (GenDiJet_electron_e.at(0) > -1) return sqrt(GenDiJet_electron_e*GenDiJet_electron_e - GenDiJet_electron_px*GenDiJet_electron_px - GenDiJet_electron_py*GenDiJet_electron_py - GenDiJet_electron_pz*GenDiJet_electron_pz).at(0); else return float(-1.);")
  

                 # Define Di-jet - positron (e+) 4 Vect
                #.Define("Gen_positron4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenHNL_positron_pt, GenHNL_positron_eta, GenHNL_positron_phi, GenHNL_positron_e)")
                #.Define("GenDiJet_positron4Vect", "ReconstructedParticle::get_tlv_sum(Gen_positron4Vect, GenDiJet)")
                #.Define("GenDiJet_positron_e", "ReconstructedParticle::get_tlv_e(GenDiJet_positron4Vect)")
                #.Define("GenDiJet_positron_theta", "if (GenDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(GenDiJet_positron4Vect).at(0); else return float(1000.);")
                #.Define("GenDiJet_positron_px", "if (GenDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(GenDiJet_positron4Vect).at(0); else return float(-3000.);")
                #.Define("GenDiJet_positron_py", "if (GenDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(GenDiJet_positron4Vect).at(0); else return float(-1000.);")
                #.Define("GenDiJet_positron_pz", "if (GenDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(GenDiJet_positron4Vect).at(0); else return float(-1000.);")
                #.Define("GenDiJet_positron_invMass", "if (GenDiJet_positron_e.at(0) > -1) return sqrt(GenDiJet_positron_e*GenDiJet_positron_e - GenDiJet_positron_px*GenDiJet_positron_px - GenDiJet_positron_py*GenDiJet_positron_py - GenDiJet_positron_pz*GenDiJet_positron_pz).at(0); else return float(-1.);")

                # Finding the Lxy of the HNL
                # Definition: Lxy = math.sqrt( (branchGenPtcl.At(daut1).X)**2 + (branchGenPtcl.At(daut1).Y)**2 )
                #.Define("GenHNL_Lxy", "return sqrt(GenHNLElectron_vertex_x*GenHNLElectron_vertex_x + GenHNLElectron_vertex_y*GenHNLElectron_vertex_y)")
                # Finding the Lxyz of the HNL
                #.Define("GenHNL_Lxyz", "return sqrt(GenHNLElectron_vertex_x*GenHNLElectron_vertex_x + GenHNLElectron_vertex_y*GenHNLElectron_vertex_y + GenHNLElectron_vertex_z*GenHNLElectron_vertex_z)")

                #Differences in theta between DiJet and HNL (gen-level)
                #.Define("GenHNL_DiJet_Delta_theta", "return atan2(sin(GenDiJet_theta - GenHNL_theta), cos(GenDiJet_theta - GenHNL_theta))")
             
                #Diff in theta between HNL and Electron (e+ and e-)
                #.Define("GenHNL_Electron_Delta_theta", "return atan2(sin(GenHNLElectron_theta - GenHNL_theta), cos(GenHNLElectron_theta - GenHNL_theta))")
               
                #Cos(theta) between HNL and Electron
                #.Define("GenHNL_Electron_cos_theta", "return cos(GenHNL_Electron_Delta_theta)")
 
                #Diff in theta between HNL and electron (e-) 
                #.Define("GenHNLelectron_Delta_theta", "if (GenHNL_electron_theta > -100) return atan2(sin(GenHNL_electron_theta - GenHNL_theta), cos(GenHNL_electron_theta - GenHNL_theta)).at(0); else return float (-900.);")

                #Diff in theta between HNL and positron (e+)
                #.Define("GenHNLpositron_Delta_theta", "if (GenHNL_positron_theta > -100) return atan2(sin(GenHNL_positron_theta - GenHNL_theta), cos(GenHNL_positron_theta - GenHNL_theta)).at(0); else return float(-800.);")

                #Diff in theta between DiJet and Electron (e+ and e-)
                #.Define("GenHNLElectron_DiJet_Delta_theta", "return atan2(sin(GenDiJet_theta - GenHNLElectron_theta), cos(GenDiJet_theta - GenHNLElectron_theta))")
                #Diff in theta between DiJet and electron (e-)
                #.Define("GenHNL_electron_DiJet_Delta_theta", "if (GenHNL_electron_theta > -100) return atan2(sin(GenDiJet_theta - GenHNL_electron_theta), cos(GenDiJet_theta - GenHNL_electron_theta)); else return float(600.);")
                #Diff in theta between DiJet and positron (e+)
                #.Define("GenHNL_positron_DiJet_Delta_theta", "if (GenHNL_positron_theta > -100) return atan2(sin(GenDiJet_theta - GenHNL_positron_theta), cos(GenDiJet_theta - GenHNL_positron_theta)); else return float(500.);")
                # Diff in theta between DiJet-Electron and Electron (e+ and e-)
                #.Define("GenDiJetElectron_Electron_Delta_theta", "return atan2(sin(GenDiJetElectron_theta - GenHNLElectron_theta), cos(GenDiJetElectron_theta - GenHNLElectron_theta))") 
               
                # Diff in theta between DiJet-Electron and electron (e-)
                #.Define("GenDiJet_electron_electron_Delta_theta", "if (GenHNL_electron_theta > -100) return atan2(sin(GenDiJet_electron_theta - GenHNL_electron_theta), cos(GenDiJet_electron_theta - GenHNL_electron_theta)); else return float(400.);")
                # Diff in theta between DiJet-Electron and positron (e+)
                #.Define("GenDiJet_positron_positron_Delta_theta", "if (GenHNL_positron_theta > -100) return atan2(sin(GenDiJet_positron_theta - GenHNL_positron_theta), cos(GenDiJet_positron_theta - GenHNL_positron_theta)); else return float(300.);") 

                # Defining diff between lead jet and prompt electron
                #.Define("LeadJet_HNLELectron_Delta_e", "return GenHNLElectron_e - RecoLeadJet_e")
                #.Define("LeadJet_HNLELectron_Delta_pt", "return (GenHNLElectron_pt - RecoLeadJet_pt)")
                #.Define("LeadJet_HNLELectron_Delta_phi", "if (RecoLeadJet_phi < -1000) return float(200.); else return atan2(sin(RecoLeadJet_phi - GenHNLElectron_phi), cos(RecoLeadJet_phi - GenHNLElectron_phi));")
                #.Define("LeadJet_HNLELectron_Delta_eta", "return (RecoLeadJet_eta - GenHNLElectron_eta)")
                #.Define("LeadJet_HNLELectron_Delta_R", "return sqrt(LeadJet_HNLELectron_Delta_phi*LeadJet_HNLELectron_Delta_phi + LeadJet_HNLELectron_Delta_eta*LeadJet_HNLELectron_Delta_eta)")

                #Define Di-Jet 4vector
                .Define("LeadingJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoLeadJet_pt, RecoLeadJet_eta, RecoLeadJet_phi, RecoLeadJet_e)")
                .Define("SecondJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoSecondJet_pt, RecoSecondJet_eta, RecoSecondJet_phi, RecoSecondJet_e)")
                .Define("RecoDiJet", "ReconstructedParticle::get_tlv_sum(LeadingJet4Vect, SecondJet4Vect)")
                .Define("RecoDiJet_e", "return ReconstructedParticle::get_tlv_e(RecoDiJet).at(0)")
                .Define("RecoDiJet_phi", "if (RecoDiJet_e > -1) return ReconstructedParticle::get_tlv_phi(RecoDiJet).at(0); else return float(1000.);")
                .Define("RecoDiJet_pt", "if (RecoDiJet_e > -1) return ReconstructedParticle::get_tlv_pt(RecoDiJet).at(0); else return float(1000.);")
                .Define("RecoDiJet_eta", "if (RecoDiJet_e > -1) return ReconstructedParticle::get_tlv_eta(RecoDiJet).at(0); else return float(1000.);")
                .Define("RecoDiJet_theta", "if (RecoDiJet_e > -1) return ReconstructedParticle::get_tlv_theta(RecoDiJet).at(0); else return float(1000.);")
                .Define("RecoDiJet_px", "if (RecoDiJet_e > -1) return ReconstructedParticle::get_tlv_px(RecoDiJet).at(0); else return float(-500.);")
                .Define("RecoDiJet_py", "if (RecoDiJet_e > -1) return ReconstructedParticle::get_tlv_py(RecoDiJet).at(0); else return float(-1000.);")
                .Define("RecoDiJet_pz", "if (RecoDiJet_e > -1) return ReconstructedParticle::get_tlv_pz(RecoDiJet).at(0); else return float(-1000.);")
                .Define("RecoDiJet_invMass", "if (RecoDiJet_e > -1) return sqrt(RecoDiJet_e*RecoDiJet_e - RecoDiJet_px*RecoDiJet_px - RecoDiJet_py*RecoDiJet_py - RecoDiJet_pz*RecoDiJet_pz); else return float(-1.);")

                #.Define("DiJet_HNLElectron_Delta_e", "if (RecoDiJet_e > -1) return GenHNLElectron_e - RecoDiJet_e; else return float(1000.);")
                #.Define("DiJet_HNLElectron_Delta_pt", "return (GenHNLElectron_pt - RecoDiJet_pt)")
                #.Define("DiJet_HNLElectron_Delta_phi", "if (RecoDiJet_phi > 500) return float(-100.); else return atan2(sin(RecoDiJet_phi - GenHNLElectron_phi), cos(RecoDiJet_phi - GenHNLElectron_phi));")
                #.Define("DiJet_HNLElectron_Delta_eta", "return(GenHNLElectron_eta - RecoDiJet_eta)")
                #.Define("DiJet_HNLElectron_Delta_R", "return sqrt(DiJet_HNLElectron_Delta_phi*DiJet_HNLElectron_Delta_phi + DiJet_HNLElectron_Delta_eta*DiJet_HNLElectron_Delta_eta)")
              
                # Reconstructed variables
                # Electrons
                .Define("RecoElectron_e",      "ReconstructedParticle::get_e(RecoElectrons)")
                .Define("RecoElectron_p",      "ReconstructedParticle::get_p(RecoElectrons)")
                .Define("RecoElectron_pt",      "ReconstructedParticle::get_pt(RecoElectrons)")
                .Define("RecoElectron_px",      "ReconstructedParticle::get_px(RecoElectrons)")
                .Define("RecoElectron_py",      "ReconstructedParticle::get_py(RecoElectrons)")
                .Define("RecoElectron_pz",      "ReconstructedParticle::get_pz(RecoElectrons)")
		.Define("RecoElectron_eta",     "ReconstructedParticle::get_eta(RecoElectrons)") #pseudorapidity eta
                .Define("RecoElectron_theta",   "ReconstructedParticle::get_theta(RecoElectrons)")
		.Define("RecoElectron_phi",     "ReconstructedParticle::get_phi(RecoElectrons)") #polar angle in the transverse plane phi
                .Define("RecoElectron_charge",  "ReconstructedParticle::get_charge(RecoElectrons)")
                .Define("RecoElectronTrack_absD0", "return abs(ReconstructedParticle2Track::getRP2TRK_D0(RecoElectrons,EFlowTrack_1))")
                .Define("RecoElectronTrack_absZ0", "return abs(ReconstructedParticle2Track::getRP2TRK_Z0(RecoElectrons,EFlowTrack_1))")
                .Define("RecoElectronTrack_absD0sig", "return abs(ReconstructedParticle2Track::getRP2TRK_D0_sig(RecoElectrons,EFlowTrack_1))") #significance
                .Define("RecoElectronTrack_absZ0sig", "return abs(ReconstructedParticle2Track::getRP2TRK_Z0_sig(RecoElectrons,EFlowTrack_1))")
                .Define("RecoElectronTrack_D0cov", "ReconstructedParticle2Track::getRP2TRK_D0_cov(RecoElectrons,EFlowTrack_1)") #variance (not sigma)
                .Define("RecoElectronTrack_Z0cov", "ReconstructedParticle2Track::getRP2TRK_Z0_cov(RecoElectrons,EFlowTrack_1)")

                .Define("RecoElectronTracks",   "ReconstructedParticle2Track::getRP2TRK( RecoElectrons, EFlowTrack_1)")
              
                .Define("RecoElectron_lead_e", "if (n_RecoElectrons >=1) return ReconstructedParticle::get_e(RecoElectrons).at(0); else return float(-1.);")
                .Define("RecoElectron_lead_pt", "if (n_RecoElectrons >=1) return ReconstructedParticle::get_pt(RecoElectrons).at(0); else return float(-1.);")
                .Define("RecoElectron_lead_eta", "if (n_RecoElectrons >=1) return ReconstructedParticle::get_eta(RecoElectrons).at(0); else return float(-100.);")
                .Define("RecoElectron_lead_phi", "if (n_RecoElectrons >=1) return ReconstructedParticle::get_phi(RecoElectrons).at(0); else return float(-100.);")
                .Define("RecoElectron_4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoElectron_lead_pt, RecoElectron_lead_eta, RecoElectron_lead_phi, RecoElectron_lead_e)")
                .Define("RecoElectron_4Vect_e", "return ReconstructedParticle::get_tlv_e(RecoElectron_4Vect).at(0)")
                .Define("RecoHNLElectron_theta", "if (RecoElectron_lead_e > -1) return ReconstructedParticle::get_tlv_theta(RecoElectron_4Vect).at(0); else return float(-100.)") 
                # electrons (-)
                .Define("RecoHNL_electron_e", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) < 0) return RecoElectron_e.at(0); else return float(-1.);")
                .Define("RecoHNL_electron_pt", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) < 0) return RecoElectron_pt.at(0); else return float(-1.);")
                .Define("RecoHNL_electron_eta", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) < 0) return RecoElectron_eta.at(0); else return float(-100.);")
                .Define("RecoHNL_electron_phi", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) < 0) return RecoElectron_phi.at(0); else return float(-100.);") 
                .Define("RecoHNL_electron_theta", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) < 0) return RecoElectron_theta.at(0); else return float(-100.);") 
                .Define("RecoHNL_electron_4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoHNL_electron_pt, RecoHNL_electron_eta, RecoHNL_electron_phi, RecoHNL_electron_e)")

                # positrons (+)
                .Define("RecoHNL_positron_e", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) > 0) return RecoElectron_e.at(0); else return float(-1.);")
                .Define("RecoHNL_positron_pt", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) > 0) return RecoElectron_pt.at(0); else return float(-1.);")
                .Define("RecoHNL_positron_eta", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) > 0) return RecoElectron_eta.at(0); else return float(-100.);")
                .Define("RecoHNL_positron_phi", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) > 0) return RecoElectron_phi.at(0); else return float(-100.);")
                .Define("RecoHNL_positron_theta", "if (n_RecoElectrons >= 1 && RecoElectron_charge.at(0) > 0) return RecoElectron_theta.at(0); else return float(-100.);")
                .Define("RecoHNL_positron_4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoHNL_positron_pt, RecoHNL_positron_eta, RecoHNL_positron_phi, RecoHNL_positron_e)")

                #Build Delta phi/eta between variables (electron, jets)
                .Define("RecoElectron_LeadJet_delta_phi", "if (RecoLeadJet_phi < -10) return float(1000.); else return atan2(sin(RecoLeadJet_phi - RecoElectron_lead_phi), cos(RecoLeadJet_phi - RecoElectron_lead_phi));")
                .Define("RecoElectron_SecondJet_delta_phi", "if (RecoSecondJet_phi < -10) return float(1000.); else return atan2(sin(RecoSecondJet_phi - RecoElectron_lead_phi), cos(RecoSecondJet_phi - RecoElectron_lead_phi));")
                .Define("RecoElectron_DiJet_delta_phi", "if (RecoDiJet_phi > 100) return float(1000.); else return atan2(sin(RecoDiJet_phi - RecoElectron_lead_phi), cos(RecoDiJet_phi - RecoElectron_lead_phi));")
                .Define("RecoDiJet_delta_phi", "if (RecoLeadJet_phi < -10) return float(1000.); else return atan2(sin(RecoLeadJet_phi - RecoSecondJet_phi), cos(RecoLeadJet_phi - RecoSecondJet_phi));")

                .Define("RecoElectron_LeadJet_delta_eta", "if (RecoLeadJet_eta < -10) return float(1000.); else return atan2(sin(RecoLeadJet_eta - RecoElectron_lead_eta), cos(RecoLeadJet_eta - RecoElectron_lead_eta));")
                .Define("RecoElectron_SecondJet_delta_eta", "if (RecoSecondJet_eta < -10) return float(1000.); else return atan2(sin(RecoSecondJet_eta - RecoElectron_lead_eta), cos(RecoSecondJet_eta - RecoElectron_lead_eta));")
                .Define("RecoElectron_DiJet_delta_eta", "if (RecoDiJet_eta > 10) return float(1000.); else return atan2(sin(RecoDiJet_eta - RecoElectron_lead_eta), cos(RecoDiJet_eta - RecoElectron_lead_eta));")
                .Define("RecoDiJet_delta_eta", "if (RecoLeadJet_eta < -10) return float(1000.); else return atan2(sin(RecoLeadJet_eta - RecoSecondJet_eta), cos(RecoLeadJet_eta - RecoSecondJet_eta));")

                #Build Delta_R between electron and jets:
                .Define("RecoElectron_LeadJet_delta_R", "return sqrt(RecoElectron_LeadJet_delta_phi*RecoElectron_LeadJet_delta_phi + RecoElectron_LeadJet_delta_eta*RecoElectron_LeadJet_delta_eta)") 
                .Define("RecoElectron_SecondJet_delta_R", "return sqrt(RecoElectron_SecondJet_delta_phi*RecoElectron_SecondJet_delta_phi + RecoElectron_SecondJet_delta_eta*RecoElectron_SecondJet_delta_eta)") 
                .Define("RecoElectron_DiJet_delta_R", "return sqrt(RecoElectron_DiJet_delta_phi*RecoElectron_DiJet_delta_phi + RecoElectron_DiJet_delta_eta*RecoElectron_DiJet_delta_eta)")         
                .Define("RecoDiJet_delta_R", "return sqrt(RecoDiJet_delta_phi*RecoDiJet_delta_phi + RecoDiJet_delta_eta*RecoDiJet_delta_eta)")       




                # Build DiJet + Electron 4 Vect (Reco-level)
                .Define("RecoDiJetElectron4Vect", "ReconstructedParticle::get_tlv_sum(RecoElectron_4Vect, RecoDiJet)")
                .Define("RecoDiJetElectron_e", "return ReconstructedParticle::get_tlv_e(RecoDiJetElectron4Vect).at(0)")
                .Define("RecoDiJetElectron_pt", "if (RecoDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_pt(RecoDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJetElectron_eta", "if (RecoDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_eta(RecoDiJetElectron4Vect).at(0); else return float(1000.);")
                .Define("RecoDiJetElectron_phi", "if (RecoDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_phi(RecoDiJetElectron4Vect).at(0); else return float(1000.);")
                .Define("RecoDiJetElectron_theta", "if (RecoDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_theta(RecoDiJetElectron4Vect).at(0); else return float(1000.);")
                .Define("RecoDiJetElectron_px", "if (RecoDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_px(RecoDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJetElectron_py", "if (RecoDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_py(RecoDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJetElectron_pz", "if (RecoDiJetElectron_e > -1) return ReconstructedParticle::get_tlv_pz(RecoDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJetElectron_invMass", "if (RecoDiJetElectron_e > -1) return sqrt(RecoDiJetElectron_e*RecoDiJetElectron_e - RecoDiJetElectron_px*RecoDiJetElectron_px - RecoDiJetElectron_py*RecoDiJetElectron_py - RecoDiJetElectron_pz*RecoDiJetElectron_pz); else return float(-1.);")

 
                # Build DiJet + e(-) 4 Vect (Reco-level)
                .Define("RecoDiJet_electron4Vect", "ReconstructedParticle::get_tlv_sum(RecoHNL_electron_4Vect, RecoDiJet)")
                .Define("RecoDiJet_electron_e", "ReconstructedParticle::get_tlv_e(RecoDiJet_electron4Vect)")
                .Define("RecoDiJet_electron_theta", "if (RecoDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(RecoDiJet_electron4Vect).at(0); else return float(1000.);")
                .Define("RecoDiJet_electron_px", "if (RecoDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(RecoDiJet_electron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJet_electron_py", "if (RecoDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(RecoDiJet_electron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJet_electron_pz", "if (RecoDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(RecoDiJet_electron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJet_electron_invMass", "if (RecoDiJet_electron_e.at(0) > -1) return sqrt(RecoDiJet_electron_e*RecoDiJet_electron_e - RecoDiJet_electron_px*RecoDiJet_electron_px - RecoDiJet_electron_py*RecoDiJet_electron_py - RecoDiJet_electron_pz*RecoDiJet_electron_pz).at(0); else return float(-1.);")

                
                # Build DiJet + e(+) 4 Vect (Reco-level)
                .Define("RecoDiJet_positron4Vect", "ReconstructedParticle::get_tlv_sum(RecoHNL_positron_4Vect, RecoDiJet)")
                .Define("RecoDiJet_positron_e", "ReconstructedParticle::get_tlv_e(RecoDiJet_positron4Vect)")
                .Define("RecoDiJet_positron_theta", "if (RecoDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(RecoDiJet_positron4Vect).at(0); else return float(1000.);")
                .Define("RecoDiJet_positron_px", "if (RecoDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(RecoDiJet_positron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJet_positron_py", "if (RecoDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(RecoDiJet_positron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJet_positron_pz", "if (RecoDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(RecoDiJet_positron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJet_positron_invMass", "if (RecoDiJet_positron_e.at(0) > -1) return sqrt(RecoDiJet_positron_e*RecoDiJet_positron_e - RecoDiJet_positron_px*RecoDiJet_positron_px - RecoDiJet_positron_py*RecoDiJet_positron_py - RecoDiJet_positron_pz*RecoDiJet_positron_pz).at(0); else return float(-1.);")

                # Now we reconstruct the reco decay vertex using the reco'ed tracks from electrons
                # First the full object, of type Vertexing::FCCAnalysesVertex
                .Define("RecoDecayVertexObject", "VertexFitterSimple::VertexFitter_Tk( 2, RecoElectronTracks)" )

                # from which we extract the edm4hep::VertexData object, which contains the vertex position in mm
                .Define("RecoDecayVertex",  "VertexingUtils::get_VertexData( RecoDecayVertexObject )")

                .Define("Reco_Lxy", "return sqrt(RecoDecayVertex.position.x*RecoDecayVertex.position.x + RecoDecayVertex.position.y*RecoDecayVertex.position.y)")
                .Define("Reco_Lxyz","return sqrt(RecoDecayVertex.position.x*RecoDecayVertex.position.x + RecoDecayVertex.position.y*RecoDecayVertex.position.y + RecoDecayVertex.position.z*RecoDecayVertex.position.z)")

                # Compute delta_theta at reco-level
                #Diff in theta between DiJet and Electron (e+ and e-)
                .Define("RecoHNLElectron_DiJet_Delta_theta", "return atan2(sin(RecoDiJet_theta - RecoHNLElectron_theta), cos(RecoDiJet_theta - RecoHNLElectron_theta))")
                #Diff in theta between DiJet and electron (e-)
                .Define("RecoHNL_electron_DiJet_Delta_theta", "if (RecoHNL_electron_theta > -100) return atan2(sin(RecoDiJet_theta - RecoHNL_electron_theta), cos(RecoDiJet_theta - RecoHNL_electron_theta)); else return float(-200.);")
                #Diff in theta between DiJet and positron (e+)
                .Define("RecoHNL_positron_DiJet_Delta_theta", "if (RecoHNL_positron_theta > -100) return atan2(sin(RecoDiJet_theta - RecoHNL_positron_theta), cos(RecoDiJet_theta - RecoHNL_positron_theta)); else return float(-300.);")
                # Diff in theta between DiJet-Electron and Electron (e+ and e-)
                .Define("RecoDiJetElectron_Electron_Delta_theta", "return atan2(sin(RecoDiJetElectron_theta - RecoHNLElectron_theta), cos(RecoDiJetElectron_theta - RecoHNLElectron_theta))")

                # Diff in theta between DiJet-Electron and electron (e-)
                .Define("RecoDiJet_electron_electron_Delta_theta", "if (RecoHNL_electron_theta > -100) return atan2(sin(RecoDiJet_electron_theta - RecoHNL_electron_theta), cos(RecoDiJet_electron_theta - RecoHNL_electron_theta)); else return float(-400.);")
                # Diff in theta between DiJet-Electron and positron (e+)
                .Define("RecoDiJet_positron_positron_Delta_theta", "if (RecoHNL_positron_theta > -100) return atan2(sin(RecoDiJet_positron_theta - RecoHNL_positron_theta), cos(RecoDiJet_positron_theta - RecoHNL_positron_theta)); else return float(-500.);")


                .Define("RecoPhoton_e",      "ReconstructedParticle::get_e(RecoPhotons)")
                .Define("RecoPhoton_p",      "ReconstructedParticle::get_p(RecoPhotons)")
                .Define("RecoPhoton_pt",      "ReconstructedParticle::get_pt(RecoPhotons)")
                .Define("RecoPhoton_px",      "ReconstructedParticle::get_px(RecoPhotons)")
                .Define("RecoPhoton_py",      "ReconstructedParticle::get_py(RecoPhotons)")
                .Define("RecoPhoton_pz",      "ReconstructedParticle::get_pz(RecoPhotons)")
		.Define("RecoPhoton_eta",     "ReconstructedParticle::get_eta(RecoPhotons)") #pseudorapidity eta
                .Define("RecoPhoton_theta",   "ReconstructedParticle::get_theta(RecoPhotons)")
		.Define("RecoPhoton_phi",     "ReconstructedParticle::get_phi(RecoPhotons)") #polar angle in the transverse plane phi
                .Define("RecoPhoton_charge",  "ReconstructedParticle::get_charge(RecoPhotons)")

                .Define("RecoMuon_e",      "ReconstructedParticle::get_e(RecoMuons)")
                .Define("RecoMuon_p",      "ReconstructedParticle::get_p(RecoMuons)")
                .Define("RecoMuon_pt",      "ReconstructedParticle::get_pt(RecoMuons)")
                .Define("RecoMuon_px",      "ReconstructedParticle::get_px(RecoMuons)")
                .Define("RecoMuon_py",      "ReconstructedParticle::get_py(RecoMuons)")
                .Define("RecoMuon_pz",      "ReconstructedParticle::get_pz(RecoMuons)")
		.Define("RecoMuon_eta",     "ReconstructedParticle::get_eta(RecoMuons)") #pseudorapidity eta
                .Define("RecoMuon_theta",   "ReconstructedParticle::get_theta(RecoMuons)")
		.Define("RecoMuon_phi",     "ReconstructedParticle::get_phi(RecoMuons)") #polar angle in the transverse plane phi
                .Define("RecoMuon_charge",  "ReconstructedParticle::get_charge(RecoMuons)")
                .Define("RecoMuonTrack_absD0", "return abs(ReconstructedParticle2Track::getRP2TRK_D0(RecoMuons,EFlowTrack_1))")
                .Define("RecoMuonTrack_absZ0", "return abs(ReconstructedParticle2Track::getRP2TRK_Z0(RecoMuons,EFlowTrack_1))")
                .Define("RecoMuonTrack_absD0sig", "return abs(ReconstructedParticle2Track::getRP2TRK_D0_sig(RecoMuons,EFlowTrack_1))") #significance
                .Define("RecoMuonTrack_absZ0sig", "return abs(ReconstructedParticle2Track::getRP2TRK_Z0_sig(RecoMuons,EFlowTrack_1))")
                .Define("RecoMuonTrack_D0cov", "ReconstructedParticle2Track::getRP2TRK_D0_cov(RecoMuons,EFlowTrack_1)") #variance (not sigma)
                .Define("RecoMuonTrack_Z0cov", "ReconstructedParticle2Track::getRP2TRK_Z0_cov(RecoMuons,EFlowTrack_1)")

                #EVENTWIDE VARIABLES: Access quantities that exist only once per event, such as the missing energy (despite the name, the MissingET collection contains the total missing energy)
		.Define("RecoMissingEnergy_e", "ReconstructedParticle::get_e(MissingET)")
		.Define("RecoMissingEnergy_p", "ReconstructedParticle::get_p(MissingET)")
		.Define("RecoMissingEnergy_pt", "ReconstructedParticle::get_pt(MissingET)")
		.Define("RecoMissingEnergy_px", "ReconstructedParticle::get_px(MissingET)") #x-component of RecoMissingEnergy
		.Define("RecoMissingEnergy_py", "ReconstructedParticle::get_py(MissingET)") #y-component of RecoMissingEnergy
		.Define("RecoMissingEnergy_pz", "ReconstructedParticle::get_pz(MissingET)") #z-component of RecoMissingEnergy
		.Define("RecoMissingEnergy_eta", "ReconstructedParticle::get_eta(MissingET)")
		.Define("RecoMissingEnergy_theta", "ReconstructedParticle::get_theta(MissingET)")
		.Define("RecoMissingEnergy_phi", "ReconstructedParticle::get_phi(MissingET)") #angle of RecoMissingEnergy

                # ee invariant mass
                .Define("Reco_ee_energy", "if (n_RecoElectrons>1) return (RecoElectron_e.at(0) + RecoElectron_e.at(1)); else return float(-1.);")
                .Define("Reco_ee_px", "if (n_RecoElectrons>1) return (RecoElectron_px.at(0) + RecoElectron_px.at(1)); else return float(-1.);")
                .Define("Reco_ee_py", "if (n_RecoElectrons>1) return (RecoElectron_py.at(0) + RecoElectron_py.at(1)); else return float(-1.);")
                .Define("Reco_ee_pz", "if (n_RecoElectrons>1) return (RecoElectron_pz.at(0) + RecoElectron_pz.at(1)); else return float(-1.);")
                .Define("Reco_ee_invMass", "if (n_RecoElectrons>1) return sqrt(Reco_ee_energy*Reco_ee_energy - Reco_ee_px*Reco_ee_px - Reco_ee_py*Reco_ee_py - Reco_ee_pz*Reco_ee_pz ); else return float(-1.);")

                # minDR (reco electrons, reco jets)
                .Define("RecoElRecoJets_ee_kt_minDR", "ReconstructedParticle::minDR(RecoJets_ee_kt_pt, RecoJets_ee_kt_eta, RecoJets_ee_kt_phi, RecoElectron_pt, RecoElectron_eta, RecoElectron_phi, 10000, 0.3)")
                .Define("RecoElGenEl_minDR", "ReconstructedParticle::minDR(FSGenElectron_pt, FSGenElectron_eta, FSGenElectron_phi, RecoElectron_pt, RecoElectron_eta, RecoElectron_phi, 10000, 0.3)")
                #.Define("RecoJets_ee_ktGenJets_ee_ktminDR", "ReconstructedParticle::minDR(RecoJets_ee_kt_pt, RecoJets_ee_kt_eta, RecoJets_ee_kt_phi, GenJets_ee_ktpt, GenJets_ee_kteta, GenJets_ee_ktphi, 10000, 0.3)")
                #.Define("RecoElGenJets_ee_ktminDR", "ReconstructedParticle::minDR(RecoElectron_pt, RecoElectron_eta, RecoElectron_phi, GenJets_ee_ktpt, GenJets_ee_kteta, GenJets_ee_ktphi, 10000, 0.3)")
                .Define("GenElRecoJets_ee_kt_minDR", "ReconstructedParticle::minDR(RecoJets_ee_kt_pt, RecoJets_ee_kt_eta, RecoJets_ee_kt_phi, FSGenElectron_pt, FSGenElectron_eta, FSGenElectron_phi, 10000, 0.3)")
               )
                return df2

        def output():
                branchList = [
                        ######## Monte-Carlo particles #######
                        "n_MCParticles", 
                        "n_FSGenElectron",
                        "FSGenElectron_e",
                        "FSGenElectron_p",
                        "FSGenElectron_pt",
                        "FSGenElectron_px",
                        "FSGenElectron_py",
                        "FSGenElectron_pz",
                        "FSGenElectron_eta",
                        "FSGenElectron_theta",
                        "FSGenElectron_phi",
                        "FSGenElectron_charge",
                        "FSGenElectron_vertex_x",
                        "FSGenElectron_vertex_y",
                        "FSGenElectron_vertex_z",
                        "FSGen_Lxy",
                        "FSGen_Lxyz",
                        "n_FSGenNeutrino",
                        "FSGenNeutrino_e",
                        "FSGenNeutrino_p",
                        "FSGenNeutrino_pt",
                        "FSGenNeutrino_px",
                        "FSGenNeutrino_py",
                        "FSGenNeutrino_pz",
                        "FSGenNeutrino_eta",
                        "FSGenNeutrino_theta",
                        "FSGenNeutrino_phi",
                        "FSGenNeutrino_charge",
                        "n_FSGenPhoton",
                        "FSGenPhoton_e",
                        "FSGenPhoton_p",
                        "FSGenPhoton_pt",
                        "FSGenPhoton_px",
                        "FSGenPhoton_py",
                        "FSGenPhoton_pz",
                        "FSGenPhoton_eta",
                        "FSGenPhoton_theta",
                        "FSGenPhoton_phi",
                        "FSGenPhoton_charge",
                      
                        "GenJets_ee_kt_n",
                        "GenJets_ee_kt_e",

                        #"n_GenJets",

                        #"Gen_selectedJet_n",

                        "GenLeadJet_e",
                        "GenLeadJet_pt",
                        "GenLeadJet_eta",
                        "GenLeadJet_phi",

                        "GenSecondJet_e",
                        "GenSecondJet_pt",
                        "GenSecondJet_eta",
                        "GenSecondJet_phi",

                        #"GenJetDelta_e",
                        #"GenJetDelta_pt",
                        #"GenJetDelta_eta",
                        #"GenJetDelta_phi",
 
                        "GenDiJet_e",
                        "GenDiJet_pt",
                        "GenDiJet_eta",
                        "GenDiJet_phi",                      
                        "GenDiJet_theta",
                        "GenDiJet_invMass",

                        #"GenHNL_theta", 
 
                        #"GenHNLElectron_e",
                        #"GenHNLElectron_pt",
                        #"GenHNLElectron_eta",
                        #"GenHNLElectron_phi",
                        #"GenElectron4Vect_invMass",

                        #"GenHNL_positron_e",
                        #"GenHNL_positron_pt",
                        #"GenHNL_positron_theta",

                        #"GenHNL_electron_e",
                        #"GenHNL_electron_pt",
                        #"GenHNL_electron_theta",
                
                        #"GenHNL_Electron_cos_theta",        
                         
                        #"GenDiJetElectron_theta",
                        #"GenDiJet_electron_theta",
                        #"GenDiJet_positron_theta",
                        #"GenDiJetElectron_invMass",
                        #"GenDiJetElectron_invMass_comp",
                        #"GenDiJet_electron_invMass",

                        #"GenDiJet_positron_invMass",

                        #"GenLeadJet_phi_e",
                        #"GenLeadJet_phi_pt",

                        #"GenSecondJet_phi_e",
                        #"GenSecondJet_phi_pt",

                        #"GenDiJetElectron_e",
                        #"GenDiJetElectron_pt",
                        #"GenDiJetElectron_eta",
                        #"GenDiJetElectron_phi",
                        

                        #"GenHNL_DiJet_Delta_theta",
                        #"GenHNL_Electron_Delta_theta",
                        #"GenHNLelectron_Delta_theta",
                        #"GenHNLpositron_Delta_theta",
                        #"GenHNLElectron_DiJet_Delta_theta",
                        #"GenHNL_electron_DiJet_Delta_theta",
                        #"GenHNL_positron_DiJet_Delta_theta",
                        #"GenDiJetElectron_Electron_Delta_theta",
                        #"GenDiJet_electron_electron_Delta_theta",
                        #"GenDiJet_positron_positron_Delta_theta",
                        
                        ######## Reconstructed particles #######
                        "n_RecoTracks",
                        "n_RecoPhotons",
                        "n_RecoElectrons",
                        "n_RecoMuons",
                        "RecoJets_ee_kt_n",
                        "RecoJets_ee_kt_e",
                        "RecoJets_ee_kt_pt",
                        "RecoJets_ee_kt_eta",
                        "RecoJets_ee_kt_theta",
                        "RecoJets_ee_kt_phi",
                        
                        "RecoLeadJet_e",
                        "RecoLeadJet_pt",
                        "RecoLeadJet_eta",
                        "RecoLeadJet_phi",
                        "RecoSecondJet_e",
                        "RecoSecondJet_pt",
                        "RecoSecondJet_eta",
                        "RecoSecondJet_phi",
  
                        "RecoJets_ee_ktDelta_e",
                        "RecoJets_ee_ktDelta_pt",
                        "RecoJets_ee_ktDelta_phi",
                        "RecoJets_ee_ktDelta_eta",
                        "RecoJets_ee_ktDelta_R",

                        "RecoElectron_LeadJet_delta_phi",
                        "RecoElectron_SecondJet_delta_phi",
                        "RecoElectron_DiJet_delta_phi",
                        "RecoDiJet_delta_phi",

                        "RecoElectron_LeadJet_delta_eta",
                        "RecoElectron_SecondJet_delta_eta",
                        "RecoElectron_DiJet_delta_eta",
                        "RecoDiJet_delta_eta",

                        "RecoElectron_LeadJet_delta_R",
                        "RecoElectron_SecondJet_delta_R",
                        "RecoElectron_DiJet_delta_R",
                        "RecoDiJet_delta_R",

                        #"LeadJet_HNLELectron_Delta_e",
                        #"LeadJet_HNLELectron_Delta_pt",
                        #"LeadJet_HNLELectron_Delta_eta",
                        #"LeadJet_HNLELectron_Delta_phi",
                        #"LeadJet_HNLELectron_Delta_R",

                        "RecoDiJet",
                        "RecoDiJet_e",
                        "RecoDiJet_pt",
                        "RecoDiJet_eta",
                        "RecoDiJet_phi",
                        "RecoDiJet_invMass",

                        "RecoDiJetElectron_invMass",
                        "RecoDiJet_electron_invMass",
                        "RecoDiJet_positron_invMass",

                        #"DiJet_HNLElectron_Delta_e",                         
                        #"DiJet_HNLElectron_Delta_pt",
                        #"DiJet_HNLElectron_Delta_phi",
                        #"DiJet_HNLElectron_Delta_eta",
                        #"DiJet_HNLElectron_Delta_R",

                        #"GenHNL_Lxy",
                        #"GenHNL_Lxyz",

                        "RecoPhoton_e",
                        "RecoPhoton_p",
                        "RecoPhoton_pt",
                        "RecoPhoton_px",
                        "RecoPhoton_py",
                        "RecoPhoton_pz",
                        "RecoPhoton_eta",
                        "RecoPhoton_theta",
                        "RecoPhoton_phi",
                        "RecoPhoton_charge",
                        "RecoElectron_e",
                        "RecoElectron_p",
                        "RecoElectron_pt",
                        "RecoElectron_px",
                        "RecoElectron_py",
                        "RecoElectron_pz",
                        "RecoElectron_eta",
                        "RecoElectron_theta",
                        "RecoElectron_phi",
                        "RecoElectron_charge",
                        "RecoElectronTrack_absD0",
                        "RecoElectronTrack_absZ0",
                        "RecoElectronTrack_absD0sig",
                        "RecoElectronTrack_absZ0sig",
                        "RecoElectronTrack_D0cov",
                        "RecoElectronTrack_Z0cov",
                        "RecoDecayVertexObject",
                        "RecoDecayVertex",

                        "RecoElectron_4Vect_e",
                        "RecoElectron_lead_e",
                        "RecoElectron_lead_pt",
                        "RecoElectron_lead_eta",
                        "RecoElectron_lead_phi",

                        "RecoHNL_electron_e", 
                        "RecoHNL_electron_pt",
                        "RecoHNL_electron_eta",
                        "RecoHNL_electron_phi",
                        "RecoHNL_electron_theta",

                        "RecoHNL_positron_e",
                        "RecoHNL_positron_pt",
                        "RecoHNL_positron_eta",
                        "RecoHNL_positron_phi",
                        "RecoHNL_positron_theta",
             
                        "RecoHNLElectron_DiJet_Delta_theta",
                        "RecoHNL_electron_DiJet_Delta_theta",
                        "RecoHNL_positron_DiJet_Delta_theta",

                        "RecoDiJetElectron_e",
                        "RecoDiJetElectron_pt",
                        "RecoDiJetElectron_eta",
                        "RecoDiJetElectron_phi",

                        "RecoDiJetElectron_px",
                        "RecoDiJetElectron_py",
                        "RecoDiJetElectron_pz",

                        "RecoDiJetElectron_theta",

                        "RecoDiJet_electron_theta", 

                        "RecoDiJet_positron_positron_Delta_theta",
                        "RecoDiJet_electron_electron_Delta_theta",
                        "RecoDiJetElectron_Electron_Delta_theta",

                        "RecoElRecoJets_ee_kt_minDR", 
                        "GenElRecoJets_ee_kt_minDR",
                        "RecoElGenEl_minDR",
                        #"RecoJets_ee_ktGenJets_ee_ktminDR",

                        "Reco_Lxy",
                        "Reco_Lxyz",
                        "RecoMuon_e",
                        "RecoMuon_p",
                        "RecoMuon_pt",
                        "RecoMuon_px",
                        "RecoMuon_py",
                        "RecoMuon_pz",
                        "RecoMuon_eta",
                        "RecoMuon_theta",
                        "RecoMuon_phi",
                        "RecoMuon_charge",
                        "RecoMuonTrack_absD0",
                        "RecoMuonTrack_absZ0",
                        "RecoMuonTrack_absD0sig",
                        "RecoMuonTrack_absZ0sig",
                        "RecoMuonTrack_D0cov",
                        "RecoMuonTrack_Z0cov", 
                        "RecoMissingEnergy_e",
                        "RecoMissingEnergy_p",
                        "RecoMissingEnergy_pt",
                        "RecoMissingEnergy_px",
                        "RecoMissingEnergy_py",
                        "RecoMissingEnergy_pz",
                        "RecoMissingEnergy_eta",
                        "RecoMissingEnergy_theta",
                        "RecoMissingEnergy_phi",

                        # enunu branches
                        "FSGen_ee_invMass",
                        "FSGen_eenu_invMass",
                        "Reco_ee_invMass",

		]

                return branchList

