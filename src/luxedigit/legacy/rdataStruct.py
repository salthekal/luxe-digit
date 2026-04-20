################################################################################################
# @info Classes for the ROOT input/output files from the various devices                       #
# @creation date 23/04/19                                                                      #
# @edit     date 23/08/06                                                                      #
#                                               more text here eventually                      #
################################################################################################
from .logger import create_logger
from datetime import datetime
import numpy as np
import ROOT
import tqdm





######################################################
######################################################
######################################################
class rdataStruct_OPT():
    # rdataStruct_FERS logger
    logging = create_logger("rdataStruct_OPT")

    def __init__(self, fname, mode="RECREATE", setVars = None):
        ## Internal
        # mode state
        self.mode = mode
        # filename
        self.ROOTfilename = fname
    
        ######################################################
        ################
        ### TTree: OPT
        ## Variable definitions
        #self.FERS = None
        self.OPT_ibunch = np.array([0], dtype=np.uint32)
        self.OPT_ievt = np.array([0], dtype=np.uint32)
        self.OPT_vdet = np.array([0, 1], dtype=np.uint32)
        self.OPT_ibunchParNb = np.array([-1], dtype=np.uint32)
        self.OPT_fcce = np.array([0], dtype=np.double)
        self.OPT_favgPairEn = np.array([0], dtype=np.double)
        self.OPT_fNoise = np.array([0], dtype=np.double)
        self.OPT_iADCres = np.array([0], dtype=np.uint32)
        self.OPT_fOlScale = np.array([0], dtype=np.double)
        self.OPT_fGain = np.array([0], dtype=np.double)
        self.OPT_vChgShrCrossTalkMap = np.zeros(10, dtype=np.double)
        #self.OPT_vparName = np.array([0,0], dtype=np.str_)
        #self.OPT_vparValue = np.array([0,0], dtype=np.double)
        #self.OPT_vparErr  = np.array([0,0], dtype=np.double)
        self.OPT_vchi2 = np.array([0, 0], dtype=np.double)
        self.OPT_vndf = np.array([0, 0], dtype=np.double)
        self.OPT_vrchi2 = np.array([0, 0], dtype=np.double)
        self.OPT_vfSA_amp = np.array([0, 0], dtype=np.double)
        self.OPT_vfSA_mea = np.array([0, 0], dtype=np.double)
        self.OPT_vfSA_sig = np.array([0, 0], dtype=np.double)
        self.OPT_vfSA_bck = np.array([0, 0], dtype=np.double)
        self.OPT_vfSA_amp_err = np.array([0, 0], dtype=np.double)
        self.OPT_vfSA_mea_err = np.array([0, 0], dtype=np.double)
        self.OPT_vfSA_sig_err = np.array([0, 0], dtype=np.double)
        self.OPT_vfSA_bck_err = np.array([0, 0], dtype=np.double)

        self.OPT_nametypes = [
            ("bunch",                   self.OPT_ibunch,                "bunch/i",                      "Bunch number"),
            ("evt",                     self.OPT_ievt,                  "evt/i",                        "Event number"),
            ("det",                     self.OPT_vdet,                  "det[2]/i",                     "Detector ID [0-1]"),          
            #       
            ('bunchParNb',              self.OPT_ibunchParNb,           "bunchParNb/i",                 "Number of particles in a bunch"),
            ('cce',                     self.OPT_fcce,                  "cce/D",                        "Charge collection efficiency [0-1]"),
            ('avgPairEn',               self.OPT_favgPairEn,            "avgPairEn/D",                  "Average pair creation energy [eV]"),
            ('fNoise',                  self.OPT_fNoise,                "fNoise/D",                     "Frontend noise [in electrons]"),
            ('iADCres',                 self.OPT_iADCres,               "iADCres/i",                    "Number of bits of the ADC"),
            ('fOlScale',                self.OPT_fOlScale,              "fOlScale/D",                   "Fullscale range of the ADC [in electrons]"),
            ('fGain',                   self.OPT_fGain,                 "fGain/D",                      "Gain (i.e., the ration between the amplified charge and the projected charge)"),
            ('vChgShrCrossTalkMap',     self.OPT_vChgShrCrossTalkMap,   "vChgShrCrossTalkMap[10]/D",    "Strip cross talk vector"),
            #
            #("parName",                 self.OPT_vparName,      "std::vector<string>",   "Fit parameter name"),
            #("parValue",                self.OPT_vparValue,     "std::vector<double>",   "Fit parameter value"),
            #("parErr",                  self.OPT_vparErr,       "std::vector<double>",   "Fit parameter error"),
            #
            ("chi2",                    self.OPT_vchi2,          "chi2[2]/D",                           "Chi square"),
            ("ndf",                     self.OPT_vndf,           "ndf[2]/D",                            "Number of degrees of freedom"),
            ("rchi2",                   self.OPT_vrchi2,         "rchi2[2]/D",                          "Reduced chisquare (chi2/ndf)"),
            ("fSA_amp",                 self.OPT_vfSA_amp,       "fSA_amp[2]/D",                        "Fit scheme A - amplitude"),
            ("fSA_mea",                 self.OPT_vfSA_mea,       "fSA_mea[2]/D",                        "Fit scheme A - mean value"),
            ("fSA_sig",                 self.OPT_vfSA_sig,       "fSA_sig[2]/D",                        "Fit scheme A - sigma"),
            ("fSA_bck",                 self.OPT_vfSA_bck,       "fSA_bck[2]/D",                        "Fit scheme A - background value"),
            ("fSA_amp_err",             self.OPT_vfSA_amp_err,   "fSA_amp_err[2]/D",                    "Fit scheme A - Error on the amplitude"),
            ("fSA_mea_err",             self.OPT_vfSA_mea_err,   "fSA_mea_err[2]/D",                    "Fit scheme A - Error on the mean value"),
            ("fSA_sig_err",             self.OPT_vfSA_sig_err,   "fSA_sig_err[2]/D",                    "Fit scheme A - Error on the sigma"),
            ("fSA_bck_err",             self.OPT_vfSA_bck_err,   "fSA_bck_err[2]/D",                    "Fit scheme A - Error on the background value"),
        ]
        self.OPT_fill_warnings = {item[0]:False for item in self.OPT_nametypes}  
        ######################################################
    

        # Open ROOT output file
        self.ROOTfile = ROOT.TFile.Open(fname, mode)
        (self.logging).debug(f"Opening {fname} in {mode} mode")
        if mode == "RECREATE":
            (self.logging).debug("Attaching branches to variables.")
            # Generate the OPT TTree
            self.setupTTree_OPT()
        else:
            # OPT
            self.OPT = self.ROOTfile.OPT
            # Set branch addresses
            self.OPT_bindBranches(self.OPT)
    
    ## Functions
    # Bind branches of the OPT TTree to variables
    def OPT_bindBranches(self, tree):
        for item in self.OPT_nametypes:
            tree.SetBranchAddress(item[0], item[1])
    # Fill tree with data
    def OPT_fill(self, **kwargs):
        if (self.OPT) is None:
            raise Exception("Tree 'data' is not initialized")
        for branch in self.OPT_nametypes:
            try:
                # Filling is based on length. If the length of the array is 1 then fill the content of the array
                # otherwise fill the array itself
                if len(branch[1]) == 1:
                    branch[1][0] = kwargs[branch[0]]
                else:
                    np.copyto(branch[1], kwargs[branch[0]])
            except KeyError:
                if not self.OPT_fill_warnings[branch[0]]:
                    (self.logging).warning(f"Leaf {branch[0]} not found in kwargs (further warnings suppressed)")
                    self.OPT_fill_warnings[branch[0]] = True
        # Fill the tree
        (self.OPT).Fill()
    # Get entry i-th of the OPT TTree (for read mode)
    def OPT_getEntry(self, i):
        """
        Description 
        
        Parameters
        ----------
            i (int): Entry number 

        Returns:
        
            bunch : 0
            evt : 1
            det : 2
            parName : 3
            parValue : 4
            parErr : 5
        """
        (self.OPT).GetEntry(i)
        return [(self.OPT_ibunch)[0], (self.OPT_ievt)[0], (self.OPT_idet)[0], (self.OPT_idir)[0], self.OPT_voptPar, self.OPT_voptParErr]
    # Setup the OPT TTree
    def setupTTree_OPT(self):
        # Define the TTree
        self.OPT = ROOT.TTree("OPT", "Digitization opt analysis data")
        # Create branches
        for item in (self.OPT_nametypes):
            (self.OPT).Branch(item[0], item[1], item[2])
        # Set leaves description
        self.TREE_SetLeavesDescriptions(self.OPT, self.OPT_nametypes)

    
    
    ######################################################
    ######################################################
    ## Utilities
    # Set description (better readibility)
    def TREE_SetLeavesDescriptions(self, tree, tree_nametypes):
        for entry in tree_nametypes:
            tree.GetBranch(entry[0]).SetTitle(entry[3])

    # Write all the content of the output root file and clear datastructures for the next one
    def closeROOT(self):
        if self.mode != "READ":
            (self.OPT).Write()
        # Single file
        if type(self.ROOTfilename) is not list:
            (self.ROOTfile).Close()






######################################################
######################################################
######################################################

# Test methods
if __name__ == "__main__":
    # rdataStruct_FERS logger
    logging = create_logger("test_rdataStruct", 10)
    
    
    # Test the MERGE class
    logging.info("Testing rdataStructMERGE class by generating a testMERGE.root in this dir...")
    roFile = rdataStruct_OPT("testBuddyPLAIN.root")
    logging.info("Fill 1 without all kwargs")
    roFile.OPT_fill(bunch = 0, evt = 0)
    logging.info("Fill 2 without all kwargs")
    roFile.OPT_fill(bunch = 0, evt = 1)
    logging.info("Fill 2 witg all kwargs")
    roFile.OPT_fill(bunch = 0, evt = 2, det=0, dir=0, optPar=np.array([0,1,2]), optParErr = np.array([0,1,2]))
    roFile.closeROOT()
    logging.info("File closed.")
