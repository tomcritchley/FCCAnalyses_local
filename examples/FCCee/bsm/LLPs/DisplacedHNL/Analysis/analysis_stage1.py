#Mandatory: List of processes
processList = {

        #centrally-produced backgrounds
        'p8_ee_Zee_ecm91':{'chunks':100},
        'p8_ee_Zbb_ecm91':{'chunks':100},
        'p8_ee_Ztautau_ecm91':{'chunks':100},
        'p8_ee_Zuds_ecm91':{'chunks':100},
        'p8_ee_Zcc_ecm91':{'chunks':100},

        #privately-produced signals
        #'eenu_30GeV_1p41e-6Ve':{},
        #'eenu_50GeV_1p41e-6Ve':{},
        #'eenu_70GeV_1p41e-6Ve':{},
        #'eenu_90GeV_1p41e-6Ve':{},

        #test
        #'p8_ee_Zee_ecm91':{'fraction':0.000001},
        #'p8_ee_Zuds_ecm91':{'chunks':10,'fraction':0.000001},
}

#Production tag. This points to the yaml files for getting sample statistics
#Mandatory when running over EDM4Hep centrally produced events
#Comment out when running over privately produced events
prodTag     = "FCCee/spring2021/IDEA/"

#Input directory
#Comment out when running over centrally produced events
#Mandatory when running over privately produced events
#inputDir = "/eos/experiment/fcc/ee/analyses/case-studies/bsm/LLPs/HNLs/HNL_eenu_MadgraphPythiaDelphes"


#Optional: output directory, default is local dir
outputDir = "outputs/HNL_Dirac_ejj_50GeV_1e-3Ve/output_stage1/"
#outputDir = "/eos/user/j/jalimena/FCCeeLLP/"
#outputDir = "output_stage1/"

HNL_id = "9990012" # Dirac
#HNL_id = "9900012" # Majorana

#Optional: ncpus, default is 4
nCPUS       = 4

#Optional running on HTCondor, default is False
#runBatch    = False
runBatch    = True

#Optional batch queue name when running on HTCondor, default is workday
batchQueue = "longlunch"

#Optional computing account when running on HTCondor, default is group_u_FCC.local_gen
compGroup = "group_u_FCC.local_gen"

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
                

                # MC event primary vertex
                .Define("MC_PrimaryVertex",  "FCCAnalyses::MCParticle::get_EventPrimaryVertex(21)( Particle )" )

                # Reconstructed particles
                .Define("n_RecoTracks","ReconstructedParticle2Track::getTK_n(EFlowTrack_1)")
                
		#JETS
		.Define("n_RecoJets", "ReconstructedParticle::get_n(Jet)") #count how many jets are in the event in total
                .Define("n_GenJets" , "ReconstructedParticle::get_n(GenJet)") # Count number of jets per event (gen level)
		#PHOTONS
		.Alias("Photon0", "Photon#0.index") 
		.Define("RecoPhotons",  "ReconstructedParticle::get(Photon0, ReconstructedParticles)")
		.Define("n_RecoPhotons",  "ReconstructedParticle::get_n(RecoPhotons)") #count how many photons are in the event in total

		#ELECTRONS AND MUONS
		.Alias("Electron0", "Electron#0.index")
		.Define("RecoElectrons",  "ReconstructedParticle::get(Electron0, ReconstructedParticles)")
		.Define("n_RecoElectrons",  "ReconstructedParticle::get_n(RecoElectrons)") #count how many electrons are in the event in total

		.Alias("Muon0", "Muon#0.index")
		.Define("RecoMuons",  "ReconstructedParticle::get(Muon0, ReconstructedParticles)")
		.Define("n_RecoMuons",  "ReconstructedParticle::get_n(RecoMuons)") #count how many muons are in the event in total

		#SIMPLE VARIABLES: Access the basic kinematic variables of the (selected) jets, works analogously for electrons, muons
		.Define("RecoJet_e",      "ReconstructedParticle::get_e(Jet)")
                .Define("RecoJet_p",      "ReconstructedParticle::get_p(Jet)") #momentum p
                .Define("RecoJet_pt",      "ReconstructedParticle::get_pt(Jet)") #transverse momentum pt
                .Define("RecoJet_px",      "ReconstructedParticle::get_px(Jet)")
                .Define("RecoJet_py",      "ReconstructedParticle::get_py(Jet)")
                .Define("RecoJet_pz",      "ReconstructedParticle::get_pz(Jet)")
		.Define("RecoJet_eta",     "ReconstructedParticle::get_eta(Jet)") #pseudorapidity eta
                .Define("RecoJet_theta",   "ReconstructedParticle::get_theta(Jet)")
		.Define("RecoJet_phi",     "ReconstructedParticle::get_phi(Jet)") #polar angle in the transverse plane phi
                .Define("RecoJet_charge",  "ReconstructedParticle::get_charge(Jet)")
                .Define("RecoJetTrack_absD0", "return abs(ReconstructedParticle2Track::getRP2TRK_D0(Jet,EFlowTrack_1))")
                .Define("RecoJetTrack_absZ0", "return abs(ReconstructedParticle2Track::getRP2TRK_Z0(Jet,EFlowTrack_1))")
                .Define("RecoJetTrack_absD0sig", "return abs(ReconstructedParticle2Track::getRP2TRK_D0_sig(Jet,EFlowTrack_1))") #significance
                .Define("RecoJetTrack_absZ0sig", "return abs(ReconstructedParticle2Track::getRP2TRK_Z0_sig(Jet,EFlowTrack_1))")
                .Define("RecoJetTrack_D0cov", "ReconstructedParticle2Track::getRP2TRK_D0_cov(Jet,EFlowTrack_1)") #variance (not sigma)
                .Define("RecoJetTrack_Z0cov", "ReconstructedParticle2Track::getRP2TRK_Z0_cov(Jet,EFlowTrack_1)")

                .Define("Reco_selected_Jets", "ReconstructedParticle::sel_pt(20.)(Jet)") #select only jets with a pt > 20 GeV 
                .Define("Reco_selectedJet_e",      "ReconstructedParticle::get_e(Reco_selected_Jets)")
                .Define("Reco_selectedJet_pt",      "ReconstructedParticle::get_pt(Reco_selected_Jets)")               
                .Define("Reco_selectedJet_n", "ReconstructedParticle::get_n(Reco_selected_Jets)")
 
                # Define Gen Jet variables
                .Define("GenJet_e" ,    "ReconstructedParticle::get_e(GenJet)")
                .Define("GenJet_pt" ,    "ReconstructedParticle::get_pt(GenJet)")
                .Define("GenJet_eta" ,    "ReconstructedParticle::get_eta(GenJet)")
                .Define("GenJet_phi" ,    "ReconstructedParticle::get_phi(GenJet)")

                .Define("Gen_selected_Jets", "ReconstructedParticle::sel_pt(20.)(GenJet)") #select only jets with a pt > 20 GeV
                .Define("Gen_selectedJet_n", "ReconstructedParticle::get_n(Gen_selected_Jets)")

                # Define lead and second Gen Jet variables 
                .Define("GenLeadJet_e",  "if (n_GenJets >= 1) return (GenJet_e.at(0)); else return float(-10000.);")	 
                .Define("GenLeadJet_pt",  "if (n_GenJets >= 1) return (GenJet_pt.at(0)); else return float(-10000.);")
                .Define("GenLeadJet_eta",  "if (n_GenJets >= 1) return (GenJet_eta.at(0)); else return float(-10000.);")
                .Define("GenLeadJet_phi",  "if (n_GenJets >= 1) return (GenJet_phi.at(0)); else return float(-10000.);")

                .Define("GenSecondJet_e",  "if (n_GenJets > 1) return (GenJet_e.at(1)); else return float(-10000.);")
                .Define("GenSecondJet_pt",  "if (n_GenJets > 1) return (GenJet_pt.at(1)); else return float(-10000.);")
                .Define("GenSecondJet_eta",  "if (n_GenJets > 1) return (GenJet_eta.at(1)); else return float(-10000.);")
                .Define("GenSecondJet_phi",  "if (n_GenJets > 1) return (GenJet_phi.at(1)); else return float(-10000.);")

                # Difference between lead and secondary jet
                .Define("GenJetDelta_e", "return (GenLeadJet_e - GenSecondJet_e)")
                .Define("GenJetDelta_pt", "return (GenLeadJet_pt - GenSecondJet_pt)")
                .Define("GenJetDelta_phi", "if (GenLeadJet_phi > -1000) return atan2(sin(GenLeadJet_phi - GenSecondJet_phi), cos(GenLeadJet_phi - GenSecondJet_phi)); else return float(100.);")
                .Define("GenJetDelta_eta", "return (GenLeadJet_eta - GenSecondJet_eta)")
                .Define("GenJetDelta_R", "return sqrt(GenJetDelta_phi*GenJetDelta_phi + GenJetDelta_eta*GenJetDelta_eta)")
 

                # Building gen-level di-jet 4-vect
                .Define("GenLeadJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenLeadJet_pt, GenLeadJet_eta, GenLeadJet_phi, GenLeadJet_e)")
                .Define("GenSecondJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenSecondJet_pt, GenSecondJet_eta, GenSecondJet_phi, GenSecondJet_e)")
                .Define("GenDiJet", "ReconstructedParticle::get_tlv_sum(GenLeadJet4Vect, GenSecondJet4Vect)")

                # Define gen-level di-jet variables
                .Define("GenDiJet_e", "ReconstructedParticle::get_tlv_e(GenDiJet)")
                .Define("GenDiJet_theta", "if (GenDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(GenDiJet).at(0); else return float(-1.);")
                .Define("GenDiJet_px", "if (GenDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(GenDiJet).at(0); else return float(-1.);")
                .Define("GenDiJet_py", "if (GenDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(GenDiJet).at(0); else return float(-1.);")
                .Define("GenDiJet_pz", "if (GenDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(GenDiJet).at(0); else return float(-1.);")
                .Define("GenDiJet_invMass", "if (GenDiJet_e.at(0) > -1) return  sqrt(GenDiJet_e*GenDiJet_e - GenDiJet_px*GenDiJet_px - GenDiJet_py*GenDiJet_py - GenDiJet_pz*GenDiJet_pz).at(0); else return float(-1.);")
                # Phi analysis study
                .Define("GenLeadJet_phi_e", "if (GenLeadJet_phi > -1 && GenLeadJet_phi < 1) return float(-1.); else return GenJet_e.at(0);")
                .Define("GenLeadJet_phi_pt", "if (GenLeadJet_phi > -1 && GenLeadJet_phi < 1) return float(-1.); else return GenJet_pt.at(0);")
                
                .Define("GenSecondJet_phi_e", "if (GenSecondJet_phi > -1 && GenSecondJet_phi < 1) return float(-1.); else return GenSecondJet_e;")
                .Define("GenSecondJet_phi_pt", "if (GenSecondJet_phi > -1 && GenSecondJet_phi < 1) return float(-1.); else return GenSecondJet_pt;")

 
                # Leading Reconstructed jet variables
                .Define("RecoLeadJet_e",  "if (n_RecoJets >= 1) return (RecoJet_e.at(0)); else return float(-10000.);")
                .Define("RecoLeadJet_pt",  "if (n_RecoJets >= 1) return (RecoJet_pt.at(0)); else return float(-10000.);")
                .Define("RecoLeadJet_phi",  "if (n_RecoJets >= 1) return (RecoJet_phi.at(0)); else return float(-10000.);")
                .Define("RecoLeadJet_eta",  "if (n_RecoJets >= 1) return (RecoJet_eta.at(0)); else return float(-10000.);")
                .Define("RecoLeadJet_px", "if (n_RecoJets >= 1) return (RecoJet_px.at(0)); else return float(-1.);")
                .Define("RecoLeadJet_py", "if (n_RecoJets >= 1) return (RecoJet_py.at(0)); else return float(-1.);")
                .Define("RecoLeadJet_pz", "if (n_RecoJets >= 1) return (RecoJet_pz.at(0)); else return float(-1.);")

                # Secondary jet variables
                .Define("RecoSecondJet_e",  "if (n_RecoJets > 1) return (RecoJet_e.at(1)); else return float(-1000.);")
                .Define("RecoSecondJet_pt",  "if (n_RecoJets > 1) return (RecoJet_pt.at(1)); else return float(-1000.);")
                .Define("RecoSecondJet_phi",  "if (n_RecoJets > 1) return (RecoJet_phi.at(1)); else return float(1000.);")
                .Define("RecoSecondJet_eta",  "if (n_RecoJets > 1) return (RecoJet_eta.at(1)); else return float(1000.);")

                # Difference between lead and secondary jet
                .Define("RecoJetDelta_e", "return (RecoLeadJet_e - RecoSecondJet_e)")
                .Define("RecoJetDelta_pt", "return (RecoLeadJet_pt - RecoSecondJet_pt)")
                .Define("RecoJetDelta_phi", "if (RecoLeadJet_phi > -1000) return atan2(sin(RecoLeadJet_phi - RecoSecondJet_phi), cos(RecoLeadJet_phi - RecoSecondJet_phi)); else return float(100.);")
                .Define("RecoJetDelta_eta", "return (RecoLeadJet_eta - RecoSecondJet_eta)")
                .Define("RecoJetDelta_R", "return sqrt(RecoJetDelta_phi*RecoJetDelta_phi + RecoJetDelta_eta*RecoJetDelta_eta)")

                # Define lead jet invariant mass
                .Define("Reco_LeadJet_invMass", "if (n_RecoJets >= 1) return sqrt(RecoLeadJet_e*RecoLeadJet_e - RecoLeadJet_px*RecoLeadJet_px - RecoLeadJet_py*RecoLeadJet_py - RecoLeadJet_pz*RecoLeadJet_pz ); else return float(-1.);")

                # Defining vector containing the HNL and its daughters, in order written
                .Define("GenHNL_indices", "FCCAnalyses::MCParticle::get_indices(%s ,{11} , true, true, true, true)(Particle, Particle1)"%(HNL_id))
                .Define("GenHNL", "FCCAnalyses::MCParticle::selMC_leg(0)(GenHNL_indices, Particle)")
                .Define("GenHNL_theta", "FCCAnalyses::MCParticle::get_theta(GenHNL)")

                .Define("GenHNLElectron", "FCCAnalyses::MCParticle::selMC_leg(1)(GenHNL_indices, Particle)")
                
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
                .Define("GenHNLElectron_e", "FCCAnalyses::MCParticle::get_e(GenHNLElectron)")
                .Define("GenHNLElectron_pt", "FCCAnalyses::MCParticle::get_pt(GenHNLElectron)")
                .Define("GenHNLElectron_eta", "FCCAnalyses::MCParticle::get_eta(GenHNLElectron)")
                .Define("GenHNLElectron_phi", "FCCAnalyses::MCParticle::get_phi(GenHNLElectron)")
                .Define("GenHNLElectron_theta", "FCCAnalyses::MCParticle::get_theta(GenHNLElectron)")
                .Define("GenHNLElectron_charge", "FCCAnalyses::MCParticle::get_charge(GenHNLElectron)")
                .Define("GenHNLElectron_vertex_x","return FCCAnalyses::MCParticle::get_vertex_x(GenHNLElectron)")
                .Define("GenHNLElectron_vertex_y","return FCCAnalyses::MCParticle::get_vertex_y(GenHNLElectron)")
                .Define("GenHNLElectron_vertex_z","return FCCAnalyses::MCParticle::get_vertex_z(GenHNLElectron)")
 
                #Define electron
                .Define("GenHNL_electron_e", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_e.at(0); else return float(-1.);")
                .Define("GenHNL_electron_pt", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_pt.at(0); else return float(-1.);")
                .Define("GenHNL_electron_phi", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_phi.at(0); else return float(-100.);")
                .Define("GenHNL_electron_eta", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_eta.at(0); else return float(-100.);")
                .Define("GenHNL_electron_theta", "if (GenHNLElectron_charge.at(0) < 0) return GenHNLElectron_theta.at(0); else return float(-100.);")

                #Define positron
                .Define("GenHNL_positron_e", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_e.at(0); else return float(-1.);")
                .Define("GenHNL_positron_pt", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_pt.at(0); else return float(-1.);")
                .Define("GenHNL_positron_phi", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_phi.at(0); else return float(-100.);")
                .Define("GenHNL_positron_eta", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_eta.at(0); else return float(-100.);")
                .Define("GenHNL_positron_theta", "if (GenHNLElectron_charge.at(0) > 0) return GenHNLElectron_theta.at(0); else return float(-100.);")

                # Define Di-jet - Electron (e+ and e-) 4 Vect
                .Define("GenElectron4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenHNLElectron_pt.at(0), GenHNLElectron_eta.at(0), GenHNLElectron_phi.at(0), GenHNLElectron_e.at(0))")
                .Define("GenDiJetElectron4Vect", "ReconstructedParticle::get_tlv_sum(GenElectron4Vect, GenDiJet)")
                .Define("GenDiJetElectron_e", "ReconstructedParticle::get_tlv_e(GenDiJetElectron4Vect)")
                .Define("GenDiJetElectron_theta", "if (GenDiJetElectron_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(GenDiJetElectron4Vect).at(0); else return float(1000.);")
                .Define("GenDiJetElectron_px", "if (GenDiJetElectron_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(GenDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("GenDiJetElectron_py", "if (GenDiJetElectron_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(GenDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("GenDiJetElectron_pz", "if (GenDiJetElectron_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(GenDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("GenDiJetElectron_invMass", "if (GenDiJetElectron_e.at(0) > -1) return  sqrt(GenDiJetElectron_e*GenDiJetElectron_e - GenDiJetElectron_px*GenDiJetElectron_px - GenDiJetElectron_py*GenDiJetElectron_py - GenDiJetElectron_pz*GenDiJetElectron_pz).at(0); else return float(-1.);")

                 # Define Di-jet - electron (e-) 4 Vect
                .Define("Gen_electron4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenHNL_electron_pt, GenHNL_electron_eta, GenHNL_electron_phi, GenHNL_electron_e)")
                .Define("GenDiJet_electron4Vect", "ReconstructedParticle::get_tlv_sum(Gen_electron4Vect, GenDiJet)")
                .Define("GenDiJet_electron_e", "ReconstructedParticle::get_tlv_e(GenDiJet_electron4Vect)")
                .Define("GenDiJet_electron_theta", "if (GenDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(GenDiJet_electron4Vect).at(0); else return float(1000.);")
                .Define("GenDiJet_electron_px", "if (GenDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(GenDiJet_electron4Vect).at(0); else return float(-2000.);")
                .Define("GenDiJet_electron_py", "if (GenDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(GenDiJet_electron4Vect).at(0); else return float(-1000.);")
                .Define("GenDiJet_electron_pz", "if (GenDiJet_electron_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(GenDiJet_electron4Vect).at(0); else return float(-1000.);")
                .Define("GenDiJet_electron_invMass", "if (GenDiJet_electron_e.at(0) > -1) return sqrt(GenDiJet_electron_e*GenDiJet_electron_e - GenDiJet_electron_px*GenDiJet_electron_px - GenDiJet_electron_py*GenDiJet_electron_py - GenDiJet_electron_pz*GenDiJet_electron_pz).at(0); else return float(-1.);")
  

                 # Define Di-jet - positron (e+) 4 Vect
                .Define("Gen_positron4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(GenHNL_positron_pt, GenHNL_positron_eta, GenHNL_positron_phi, GenHNL_positron_e)")
                .Define("GenDiJet_positron4Vect", "ReconstructedParticle::get_tlv_sum(Gen_positron4Vect, GenDiJet)")
                .Define("GenDiJet_positron_e", "ReconstructedParticle::get_tlv_e(GenDiJet_positron4Vect)")
                .Define("GenDiJet_positron_theta", "if (GenDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(GenDiJet_positron4Vect).at(0); else return float(1000.);")
                .Define("GenDiJet_positron_px", "if (GenDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(GenDiJet_positron4Vect).at(0); else return float(-3000.);")
                .Define("GenDiJet_positron_py", "if (GenDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(GenDiJet_positron4Vect).at(0); else return float(-1000.);")
                .Define("GenDiJet_positron_pz", "if (GenDiJet_positron_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(GenDiJet_positron4Vect).at(0); else return float(-1000.);")
                .Define("GenDiJet_positron_invMass", "if (GenDiJet_positron_e.at(0) > -1) return sqrt(GenDiJet_positron_e*GenDiJet_positron_e - GenDiJet_positron_px*GenDiJet_positron_px - GenDiJet_positron_py*GenDiJet_positron_py - GenDiJet_positron_pz*GenDiJet_positron_pz).at(0); else return float(-1.);")

                # Finding the Lxy of the HNL
                # Definition: Lxy = math.sqrt( (branchGenPtcl.At(daut1).X)**2 + (branchGenPtcl.At(daut1).Y)**2 )
                .Define("GenHNL_Lxy", "return sqrt(GenHNLElectron_vertex_x*GenHNLElectron_vertex_x + GenHNLElectron_vertex_y*GenHNLElectron_vertex_y)")
                # Finding the Lxyz of the HNL
                .Define("GenHNL_Lxyz", "return sqrt(GenHNLElectron_vertex_x*GenHNLElectron_vertex_x + GenHNLElectron_vertex_y*GenHNLElectron_vertex_y + GenHNLElectron_vertex_z*GenHNLElectron_vertex_z)")

                #Differences in theta between DiJet and HNL (gen-level)
                .Define("GenHNL_DiJet_Delta_theta", "return atan2(sin(GenDiJet_theta - GenHNL_theta), cos(GenDiJet_theta - GenHNL_theta))")
             
                #Diff in theta between HNL and Electron (e+ and e-)
                .Define("GenHNL_Electron_Delta_theta", "return atan2(sin(GenHNLElectron_theta - GenHNL_theta), cos(GenHNLElectron_theta - GenHNL_theta))")
                
                #Diff in theta between HNL and electron (e-) 
                .Define("GenHNLelectron_Delta_theta", "if (GenHNL_electron_theta > -100) return atan2(sin(GenHNL_electron_theta - GenHNL_theta), cos(GenHNL_electron_theta - GenHNL_theta)).at(0); else return float (-900.);")

                #Diff in theta between HNL and positron (e+)
                .Define("GenHNLpositron_Delta_theta", "if (GenHNL_positron_theta > -100) return atan2(sin(GenHNL_positron_theta - GenHNL_theta), cos(GenHNL_positron_theta - GenHNL_theta)).at(0); else return float(-800.);")

                #Diff in theta between DiJet and Electron (e+ and e-)
                .Define("GenHNLElectron_DiJet_Delta_theta", "return atan2(sin(GenDiJet_theta - GenHNLElectron_theta), cos(GenDiJet_theta - GenHNLElectron_theta))")
                #Diff in theta between DiJet and electron (e-)
                .Define("GenHNL_electron_DiJet_Delta_theta", "if (GenHNL_electron_theta > -100) return atan2(sin(GenDiJet_theta - GenHNL_electron_theta), cos(GenDiJet_theta - GenHNL_electron_theta)); else return float(600.);")
                #Diff in theta between DiJet and positron (e+)
                .Define("GenHNL_positron_DiJet_Delta_theta", "if (GenHNL_positron_theta > -100) return atan2(sin(GenDiJet_theta - GenHNL_positron_theta), cos(GenDiJet_theta - GenHNL_positron_theta)); else return float(500.);")
                # Diff in theta between DiJet-Electron and Electron (e+ and e-)
                .Define("GenDiJetElectron_Electron_Delta_theta", "return atan2(sin(GenDiJetElectron_theta - GenHNLElectron_theta), cos(GenDiJetElectron_theta - GenHNLElectron_theta))") 
               
                # Diff in theta between DiJet-Electron and electron (e-)
                .Define("GenDiJet_electron_electron_Delta_theta", "if (GenHNL_electron_theta > -100) return atan2(sin(GenDiJet_electron_theta - GenHNL_electron_theta), cos(GenDiJet_electron_theta - GenHNL_electron_theta)); else return float(400.);")
                # Diff in theta between DiJet-Electron and positron (e+)
                .Define("GenDiJet_positron_positron_Delta_theta", "if (GenHNL_positron_theta > -100) return atan2(sin(GenDiJet_positron_theta - GenHNL_positron_theta), cos(GenDiJet_positron_theta - GenHNL_positron_theta)); else return float(300.);") 

                # Defining diff between lead jet and prompt electron
                .Define("LeadJet_HNLELectron_Delta_e", "return GenHNLElectron_e - RecoLeadJet_e")
                .Define("LeadJet_HNLELectron_Delta_pt", "return (GenHNLElectron_pt - RecoLeadJet_pt)")
                .Define("LeadJet_HNLELectron_Delta_phi", "if (RecoLeadJet_phi < -1000) return float(200.); else return atan2(sin(RecoLeadJet_phi - GenHNLElectron_phi.at(0)), cos(RecoLeadJet_phi - GenHNLElectron_phi.at(0)));")
                .Define("LeadJet_HNLELectron_Delta_eta", "return (RecoLeadJet_eta - GenHNLElectron_eta)")
                .Define("LeadJet_HNLELectron_Delta_R", "return sqrt(LeadJet_HNLELectron_Delta_phi*LeadJet_HNLELectron_Delta_phi + LeadJet_HNLELectron_Delta_eta*LeadJet_HNLELectron_Delta_eta)")

                .Define("LeadingJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoLeadJet_pt, RecoLeadJet_eta, RecoLeadJet_phi, RecoLeadJet_e)")
                .Define("SecondJet4Vect", "ReconstructedParticle::get_tlv_PtEtaPhiE(RecoSecondJet_pt, RecoSecondJet_eta, RecoSecondJet_phi, RecoSecondJet_e)")
                .Define("RecoDiJet", "ReconstructedParticle::get_tlv_sum(LeadingJet4Vect, SecondJet4Vect)")
                .Define("RecoDiJet_e", "ReconstructedParticle::get_tlv_e(RecoDiJet)")
                .Define("RecoDiJet_phi", "if (RecoDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_phi(RecoDiJet).at(0); else return float(1000.);")
                .Define("RecoDiJet_pt", "if (RecoDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_pt(RecoDiJet).at(0); else return float(1000.);")
                .Define("RecoDiJet_eta", "if (RecoDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_eta(RecoDiJet).at(0); else return float(1000.);")
                .Define("RecoDiJet_theta", "if (RecoDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(RecoDiJet).at(0); else return float(1000.);")
                .Define("RecoDiJet_px", "if (RecoDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(RecoDiJet).at(0); else return float(-500.);")
                .Define("RecoDiJet_py", "if (RecoDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(RecoDiJet).at(0); else return float(-1000.);")
                .Define("RecoDiJet_pz", "if (RecoDiJet_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(RecoDiJet).at(0); else return float(-1000.);")
                .Define("RecoDiJet_invMass", "if (RecoDiJet_e.at(0) > -1) return sqrt(RecoDiJet_e*RecoDiJet_e - RecoDiJet_px*RecoDiJet_px - RecoDiJet_py*RecoDiJet_py - RecoDiJet_pz*RecoDiJet_pz).at(0); else return float(-1.);")

                .Define("DiJet_HNLElectron_Delta_e", "if (RecoDiJet_e.at(0) > -1) return GenHNLElectron_e.at(0) - RecoDiJet_e.at(0); else return float(1000.);")
                .Define("DiJet_HNLElectron_Delta_pt", "return (GenHNLElectron_pt - RecoDiJet_pt)")
                .Define("DiJet_HNLElectron_Delta_phi", "if (RecoDiJet_phi > 500) return float(-100.); else return atan2(sin(RecoDiJet_phi - GenHNLElectron_phi.at(0)), cos(RecoDiJet_phi - GenHNLElectron_phi.at(0)));")
                .Define("DiJet_HNLElectron_Delta_eta", "return(GenHNLElectron_eta - RecoDiJet_eta)")
                .Define("DiJet_HNLElectron_Delta_R", "return sqrt(DiJet_HNLElectron_Delta_phi*DiJet_HNLElectron_Delta_phi + DiJet_HNLElectron_Delta_eta*DiJet_HNLElectron_Delta_eta)")
              
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

                # Build DiJet + Electron 4 Vect (Reco-level)
                .Define("RecoDiJetElectron4Vect", "ReconstructedParticle::get_tlv_sum(RecoElectron_4Vect, RecoDiJet)")
                .Define("RecoDiJetElectron_e", "ReconstructedParticle::get_tlv_e(RecoDiJetElectron4Vect)")
                .Define("RecoDiJetElectron_theta", "if (RecoDiJetElectron_e.at(0) > -1) return ReconstructedParticle::get_tlv_theta(RecoDiJetElectron4Vect).at(0); else return float(1000.);")
                .Define("RecoDiJetElectron_px", "if (RecoDiJetElectron_e.at(0) > -1) return ReconstructedParticle::get_tlv_px(RecoDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJetElectron_py", "if (RecoDiJetElectron_e.at(0) > -1) return ReconstructedParticle::get_tlv_py(RecoDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJetElectron_pz", "if (RecoDiJetElectron_e.at(0) > -1) return ReconstructedParticle::get_tlv_pz(RecoDiJetElectron4Vect).at(0); else return float(-1000.);")
                .Define("RecoDiJetElectron_invMass", "if (RecoDiJetElectron_e.at(0) > -1) return sqrt(RecoDiJetElectron_e*RecoDiJetElectron_e - RecoDiJetElectron_px*RecoDiJetElectron_px - RecoDiJetElectron_py*RecoDiJetElectron_py - RecoDiJetElectron_pz*RecoDiJetElectron_pz).at(0); else return float(-1.);")

 
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


               )
                return df2

        def output():
                branchList = [
                        ######## Monte-Carlo particles #######
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

                        "n_GenJets",
                        "GenJet_e",
                        "GenJet_pt",
                        "GenJet_eta",
                        "GenJet_phi",

                        "Gen_selectedJet_n",

                        "GenLeadJet_e",
                        "GenLeadJet_pt",
                        "GenLeadJet_eta",
                        "GenLeadJet_phi",

                        "GenSecondJet_e",
                        "GenSecondJet_pt",
                        "GenSecondJet_eta",
                        "GenSecondJet_phi",

                        "GenJetDelta_e",
                        "GenJetDelta_pt",
                        "GenJetDelta_eta",
                        "GenJetDelta_phi",
 
                        "GenDiJet_e",
                        "GenDiJet_theta",
                        "GenDiJet_invMass",

                        "GenHNL_theta", 
 
                        "GenHNLElectron_e",
                        "GenHNLElectron_pt",
                        "GenHNLElectron_eta",
                        "GenHNLElectron_phi",

                        "GenHNL_positron_e",
                        "GenHNL_positron_pt",
                        "GenHNL_positron_theta",

                        "GenHNL_electron_e",
                        "GenHNL_electron_pt",
                        "GenHNL_electron_theta",
                        
                         
                        "GenDiJetElectron_theta",
                        "GenDiJet_electron_theta",
                        "GenDiJet_positron_theta",
                        "GenDiJetElectron_invMass",

                        "GenDiJet_electron_invMass",

                        "GenDiJet_positron_invMass",

                        "GenLeadJet_phi_e",
                        "GenLeadJet_phi_pt",

                        "GenSecondJet_phi_e",
                        "GenSecondJet_phi_pt",

                        "GenHNL_DiJet_Delta_theta",
                        "GenHNL_Electron_Delta_theta",
                        "GenHNLelectron_Delta_theta",
                        "GenHNLpositron_Delta_theta",
                        "GenHNLElectron_DiJet_Delta_theta",
                        "GenHNL_electron_DiJet_Delta_theta",
                        "GenHNL_positron_DiJet_Delta_theta",
                        "GenDiJetElectron_Electron_Delta_theta",
                        "GenDiJet_electron_electron_Delta_theta",
                        "GenDiJet_positron_positron_Delta_theta",
                        
                        ######## Reconstructed particles #######
                        "n_RecoTracks",
                        "n_RecoJets",
                        "n_RecoPhotons",
                        "n_RecoElectrons",
                        "n_RecoMuons",
                        "RecoJet_e",
                        "RecoJet_p",
                        "RecoJet_pt",
                        "RecoJet_px",
                        "RecoJet_py",
                        "RecoJet_pz",
                        "RecoJet_eta",
                        "RecoJet_theta",
                        "RecoJet_phi",
                        "RecoJet_charge",
                        "RecoJetTrack_absD0",
                        "RecoJetTrack_absZ0",
                        "RecoJetTrack_absD0sig",
                        "RecoJetTrack_absZ0sig",
                        "RecoJetTrack_D0cov",
                        "RecoJetTrack_Z0cov",
                        
                        "Reco_selectedJet_n",
                        "Reco_selectedJet_e",
                        "Reco_selectedJet_pt",

                        "RecoLeadJet_e",
                        "RecoLeadJet_pt",
                        "RecoLeadJet_eta",
                        "RecoLeadJet_phi",
                        "Reco_LeadJet_invMass",
                        "RecoSecondJet_e",
                        "RecoSecondJet_pt",
                        "RecoSecondJet_eta",
                        "RecoSecondJet_phi",
  
                        "RecoJetDelta_e",
                        "RecoJetDelta_pt",
                        "RecoJetDelta_phi",
                        "RecoJetDelta_eta",
                        "RecoJetDelta_R",

                        "LeadJet_HNLELectron_Delta_e",
                        "LeadJet_HNLELectron_Delta_pt",
                        "LeadJet_HNLELectron_Delta_eta",
                        "LeadJet_HNLELectron_Delta_phi",
                        "LeadJet_HNLELectron_Delta_R",

                        "RecoDiJet",
                        "RecoDiJet_e",
                        "RecoDiJet_pt",
                        "RecoDiJet_eta",
                        "RecoDiJet_phi",
                        "RecoDiJet_invMass",

                        "RecoDiJetElectron_invMass",
                        "RecoDiJet_electron_invMass",
                        "RecoDiJet_positron_invMass",

                        "DiJet_HNLElectron_Delta_e",                         
                        "DiJet_HNLElectron_Delta_pt",
                        "DiJet_HNLElectron_Delta_phi",
                        "DiJet_HNLElectron_Delta_eta",
                        "DiJet_HNLElectron_Delta_R",

                        "GenHNL_Lxy",
                        "GenHNL_Lxyz",

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

                        "RecoElectron_lead_e",
 
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
                        "RecoDiJetElectron_theta",

                        "RecoDiJet_electron_theta", 

                        "RecoDiJet_positron_positron_Delta_theta",
                        "RecoDiJet_electron_electron_Delta_theta",
                        "RecoDiJetElectron_Electron_Delta_theta",

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
