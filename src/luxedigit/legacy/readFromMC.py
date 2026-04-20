#################################################################################################
# @info Read MC file produced by the StandaloneGBP Geant4 simualtion and prepare the bunches of #
#       signal for digitization                                                                 #
# @date 23/10/26                                                                                #
#                                                                                               #
#################################################################################################
from multipledispatch import dispatch
from .logger import create_logger
from tqdm import tqdm
import numpy as np
import ROOT
import os

# Enable ROOT implicit multithreading
ROOT.EnableImplicitMT(16)

readFromMcClassCtr = 0

class readFromMc():
    # readFromMc class logger
    logging = create_logger("readFromMc")
    
    # Electron charge in coulomb
    eCharge = 1.602176634e-19
    
    
    # self.bunches_edepMap          # list with f"b{i}_edepMapUp" and f"b{i}_edepMapDo"
    # self.profileStatErrs          # npChgDepProfile_err[det, direction, strip] = np.std(stripChgs)/np.sqrt(len(stripChgs))
    # self.bunches_chgDepProfiles   # Array of dictionaries
    # self.bunches_chgProjProfiles
    
    
    
    def __init__(self, rFname: str, bunchParNb: int, cce: float, avgPairEn: float, strictMode = True):
        """
        DD
        
        Parameters
        ----------
            rFname (str) : Filename of the MC ROOT file containing the data
            bunchParNb (int) : number of particles in a bunch
            cce (float) : Charge collection efficiency of the sensor for the bare geometrical projection of the dep chg. to proj chg.
            avgPairEn (float) : Average energy required to create an electron-hole pair in the sensor (in eV)
            strictMode (bool) : Enable checks on the CCE (by default on)
        """
        
        # Namespace introduced to ROOT overwriting
        global readFromMcClassCtr
        self.namespace = f'rFMc{int(readFromMcClassCtr)}'
        readFromMcClassCtr += 1
        
        # Initialize the internal class variables with the given ext ones.
        self.rFname = rFname
        self.bunchParNb = bunchParNb
        self.cce = cce
        self.avgPairEn = avgPairEn
        
        # Checks validity of the parameters
        if not (cce>0 and cce<1) and strictMode: raise Exception("CCE must be between 0 and 1")
        if bunchParNb <= 0: raise Exception("The number of particles in a bunch must be positive")
        if cce <=0: raise Exception("The CCE must be positive")
        if avgPairEn <=0: raise Exception("The average energy required to create an electron-hole pair must be positive")
        
        # Run the reading
        self.calcChgProjProfs()
        
        


    '''
    # Read an input ROOT file from the MC simulation and returns the maps of energy deposited in the upstream/downstream sensors
    @dispatch(int, int)
    def readEdepFromROOT(self, bunchParNb: int, pdg: int) -> list:
        """
        Read an input ROOT file from the MC simulation, where the input file is supposed to contain a certain number N*M of physical bunch simulations.
        The function then returns a tuple, with couples of maps of energy deposited in the upstream/downstream sensors.
        Errors are calculated by taking the full statistics and evaluating the std of the strip charge over the population.
        
        Parameters
        ----------
            bunchParNb (int) : number of particles (electrons or gamma) in a bunch
            pdg (int) : PDG code of the particle to consider (default: 22 = gamma) 
            
        Returns
        -------
            list indexed by the bunch nb with per item a dict: { 'bunchParNb': (int) Number particles (electrons or gamma) in the bunch,    "b0_edepMapUp": bunch 0 edepMapUp,  "b0_edepMapDo": bunch 0 edepMapDo }
        """    
        
        # If the _tmp_readEdepFromROOT.npy already exist, then read the data from the file and return immediately
        dumpFname = (self.rFname).replace('.root', f'_bunchParNb{bunchParNb}_pdg{pdg}.npy')
        try:
            msg = f"Loading {dumpFname[dumpFname.rfind('/')+1:]} from file"
            result = np.load(dumpFname, allow_pickle=True)
            msg += " with "+"\x1b[32;1m"+"success"+"\x1b[0m"
            (self.logging).debug(msg)
            return result
        except Exception as e:
            msg += f" failed ({e})"
            (self.logging).error(msg)
            (self.logging).debug("Calculating from scratch")
        
        # Open the input ROOT file
        riFile = ROOT.TFile.Open(self.rFname, "READ")
        
        # Calculate how many bunches are within this file
        PRIMARY = riFile.ntuple.PRIMARY
        # (pdg == 11) selects only electrons, in future development one may want to consider only positrons or gamma, in which case one has to use (pdg == -11 || pdg == 11) or (pdg == 22)
        primaryParNb = PRIMARY.GetEntries(f"pdg == {pdg}")
        # Number of bunches in the file
        # (bunchChg_pC*1e-12) / (1.160e-19)
        bunchNb = int(primaryParNb / bunchParNb)
        prtTypeStr = f"pdg {pdg}"
        if pdg==11:
            prtTypeStr = "electrons"
        elif pdg==22:
            prtTypeStr = "photons"
        (self.logging).debug(f"The number of bunches in the simulation is {bunchNb}. Primary {prtTypeStr} are: {prtTypeStr}")
        
        # Assert that we have non zero bunch number in the sim file
        if bunchNb == 0: raise Exception(f"There are {primaryParNb} primaries and the user number of particles in the bunch is {bunchParNb}. This gets zero particles/bunch with present MC file.")
        if bunchNb > 100: (self.logging).warning(f"Bunch number is large ({bunchNb}). Are you sure this is right?")
        
        # Alias the TTree DUTs
        DUTs = riFile.ntuple.DUTs
        
        # Create the 2D histograms
        edepMapUp_model = ROOT.TH2D("edepMapUp", "Energy deposition in the upstream sensor; X [mm]; Y [mm]; Energy deposited [keV]", 200, -10, 10, 200, -10, 10)
        edepMapUp_model.GetXaxis().CenterTitle();     edepMapUp_model.GetYaxis().CenterTitle();     edepMapUp_model.GetZaxis().CenterTitle()
        edepMapDo_model = ROOT.TH2D("edepMapDo", "Energy deposition in the downstream sensor; X [mm]; Y [mm]; Energy deposited [keV]", 200, -10, 10, 200, -10, 10) 
        edepMapDo_model.GetXaxis().CenterTitle();     edepMapDo_model.GetYaxis().CenterTitle();     edepMapDo_model.GetZaxis().CenterTitle()
        
        # Fill the histograms
        result = []
        for i in tqdm(range(bunchNb), desc="Bunch splitting", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed_s:.1f}s<>{remaining_s:.1f}s]'):
            bunch_edepMapUp = edepMapUp_model.Clone(); bunch_edepMapUp.SetName(f"b{i}_edepMapUp")
            bunch_edepMapDo = edepMapDo_model.Clone(); bunch_edepMapDo.SetName(f"b{i}_edepMapDo")
            
            DUTs.Project(f"b{i}_edepMapUp", "edepPosY:edepPosX", f"(edep) * (detID == 0) * (event >= {bunchParNb * i} && event < {bunchParNb * (i+1)})")
            DUTs.Project(f"b{i}_edepMapDo", "edepPosY:edepPosX", f"(edep) * (detID == 1) * (event >= {bunchParNb * i} && event < {bunchParNb * (i+1)})")
            
            bunch_edepMapUp.SetDirectory(0); bunch_edepMapDo.SetDirectory(0)        # This way you got the object returned correctly
            
            result.append({ f"b{i}_edepMapUp": bunch_edepMapUp, f"b{i}_edepMapDo": bunch_edepMapDo })
        
        
        # Dump the npChgDepProfile to file for future recalling
        result_np = np.array(result)
        np.save(dumpFname, result_np, allow_pickle=True)
        (self.logging).debug(f"[readEdepFromROOT] File {dumpFname[dumpFname.rfind('/')+1:]} saved on file")

        return result_np
    '''

    # Overload of readEdepFromROOT to avoid selection by pdg number (which is useless in this context)
    @dispatch(int)
    def readEdepFromROOT(self, bunchParNb: int) -> int:
        """
        Read an input ROOT file from the MC simulation, where the input file is supposed to contain a certain number N*M of physical bunch simulations.
        The function then returns a tuple, with couples of maps of energy deposited in the upstream/downstream sensors
        
        Parameters
        ----------
            bunchParNb (int) : number of particles (any type) in a bunch
            
        Returns
        -------
            bunchNb (int) : number of bunches
        """    
        
        # If the _tmp_readEdepFromROOT.npy already exist, then read the data from the file and return immediately
        bunchDumpFname = (self.rFname).replace('.root', f'_Edep_bunchParNb{bunchParNb}.npy')
        if os.path.exists(bunchDumpFname):
            (self.logging).debug(f"Loading {bunchDumpFname} from file")
            try:
                # produces something like dummyRun_100k.root -> dummyRun_100k_10_22.npy if 10 is bunchNb and 22 is pdg   
                self.bunches_edepMap = np.load(bunchDumpFname, allow_pickle=True)
                return len(self.bunches_edepMap)
            except Exception as e:
                (self.logging).error(f"Loading failed due to {e}")
        
        
        # Open the input ROOT file
        riFile = ROOT.TFile.Open(self.rFname, "READ")
        
        # Calculate how many bunches are within this file
        PRIMARY = riFile.ntuple.PRIMARY
        # (pdg == 11) selects only electrons, in future development one may want to consider only positrons or gamma, in which case one has to use (pdg == -11 || pdg == 11) or (pdg == 22)
        primaryParNb = PRIMARY.GetEntries()
        # Number of bunches in the file
        # (bunchChg_pC*1e-12) / (1.160e-19)
        bunchNb = int(primaryParNb / bunchParNb)
        (self.logging).debug(f"The number of bunches in the simulation is {bunchNb}. Primary are: {primaryParNb}")
        
        # Assert that we have non zero bunch number in the sim file
        if bunchNb == 0: raise Exception(f"There are {primaryParNb} primaries and the user number of particles in the bunch is {bunchParNb}. This gets zero particles/bunch with present MC file.")
        # Send warning if the number of bunches is large
        if bunchNb > 100: (self.logging).warning(f"Bunch number is large ({bunchNb}). Are you sure this is right?")
        
        # Alias the TTree DUTs
        DUTs = riFile.ntuple.DUTs
        # Create the 2D histograms
        edepMapUp_model = ROOT.TH2D(f"{self.namespace}edepMapUp", "Energy deposition in the upstream sensor; X [mm]; Y [mm]; Energy deposited [keV]", 200, -10, 10, 200, -10, 10)
        edepMapDo_model = ROOT.TH2D(f"{self.namespace}edepMapDo", "Energy deposition in the downstream sensor; X [mm]; Y [mm]; Energy deposited [keV]", 200, -10, 10, 200, -10, 10) 
        
        # Fill the histograms
        result = []
        for i in tqdm(range(bunchNb), desc="Bunch splitting"):
            bunch_edepMapUp = edepMapUp_model.Clone(); bunch_edepMapUp.SetName(f"{self.namespace}b{i}_edepMapUp")
            bunch_edepMapDo = edepMapDo_model.Clone(); bunch_edepMapDo.SetName(f"{self.namespace}b{i}_edepMapDo")
            
            DUTs.Project(f"{self.namespace}b{i}_edepMapUp", "edepPosY:edepPosX", f"(edep) * (detID == 0) * (event >= {bunchParNb * i} && event < {bunchParNb * (i+1)})")
            DUTs.Project(f"{self.namespace}b{i}_edepMapDo", "edepPosY:edepPosX", f"(edep) * (detID == 1) * (event >= {bunchParNb * i} && event < {bunchParNb * (i+1)})")
            
            bunch_edepMapUp.SetDirectory(0); bunch_edepMapDo.SetDirectory(0)        # This way you got the object returned correctly
            
            result.append({ f"b{i}_edepMapUp": bunch_edepMapUp, f"b{i}_edepMapDo": bunch_edepMapDo })
        
        
        # Dump the npChgDepProfile to file for future recalling
        result_np = np.array(result)
        np.save(bunchDumpFname, result_np, allow_pickle=True)
        (self.logging).debug(f"Dumped {bunchDumpFname} to file")
        
        # Store the maps in the internal variable
        self.bunches_edepMap = result_np
        
        return len(self.bunches_edepMap)


    #################################################################################################

    # Takes as input data the result of readEdepFromROOT and calculates the uncertainty to attach on each stripfor all the bunches in the run.
    def calculateProfileStatErrs(self, bunchData: list) -> np.ndarray:
        """
        Takes as input data the result of readEdepFromROOT and calculates the uncertainty to attach on each strip for all the bunches in the run.
        
        Parameters
        ----------
            bunchData (list) : list of dictionary containing the data of each bunch

        Returns
        -------
            npChgDepProfile_err (np.ndarray) : numpy array with the uncertainty on the charge deposited in each strip of each bunch
        """
        
        # Get the number of bunches
        bunchesNb = len(bunchData)
        
        # Store the profiles in a numpy array
        npChgDepProfile = np.zeros((bunchesNb, 2, 2, 200))         # bunch, det, direction, strip
        
        # Dump all the bunch projection hor/ver hist contents into a numpy array
        for bunch in range(bunchesNb):
            # Get the bunch edep 2D Map
            _edepMapUp, _edepMapDo = bunchData[bunch][f"b{bunch}_edepMapUp"], bunchData[bunch][f"b{bunch}_edepMapDo"]

            # Get only the charge deposited profile (0 entry of the result)
            chgDepProfileX_up, chgDepProfileY_up = self.fromEnergyToChargeDeposited(_edepMapUp)
            chgDepProfileX_do, chgDepProfileY_do = self.fromEnergyToChargeDeposited(_edepMapDo)
            
            binsNb = chgDepProfileX_up.GetNbinsX()
            # and store the strip charge into the  numpy array 
            for i in range(binsNb):
                npChgDepProfile[bunch, 0, 0, i] = chgDepProfileX_up.GetBinContent(i);       npChgDepProfile[bunch, 0, 1, i] = chgDepProfileY_up.GetBinContent(i)
                npChgDepProfile[bunch, 1, 0, i] = chgDepProfileX_do.GetBinContent(i);       npChgDepProfile[bunch, 1, 1, i] = chgDepProfileY_do.GetBinContent(i)
        
        # Calculates the strip std between the different bunches
        npChgDepProfile_err = np.zeros((2, 2, 200))     # det, direction, strip
        for det in [0,1]:
            for direction in [0,1]:
                for strip in range(200):
                    stripChgs = npChgDepProfile[:, det, direction, strip]
                    npChgDepProfile_err[det, direction, strip] = np.std(stripChgs)/np.sqrt(len(stripChgs))
                    
        # Return the errors (useful?)
        return npChgDepProfile_err


    #################################################################################################

    # Takes the map of energy deposited in a sensor and returns the projected horizontal and vertical profiles
    def fromEnergyToChargeDeposited(self, edepHist: ROOT.TH2D) -> tuple:
        """
        Takes the map of energy deposited in a sensor and returns the projected horizontal and vertical profiles
        
        Parameters
        ----------
            edepHist (ROOT.TH2D) : histogram with the map of energy deposition in a sensor
            
        Returns
        -------
            projProfile (tuple) : (profileX.Clone(), profileY.Clone())
        """

        # Get horizontal and vertical profiles of energy depositions in the sensor
        eDepProfileX, eDepProfileY = edepHist.ProjectionX(), edepHist.ProjectionY()
        
        # Convert the energy deposited into charge deposited
        elecFactor = 1.0
        holeFactor = 0.0 
        eDepProfileX.Scale(1000 / self.avgPairEn * (elecFactor+holeFactor) * self.eCharge)
        eDepProfileY.Scale(1000 / self.avgPairEn * (elecFactor+holeFactor) * self.eCharge)
        
        # Set the right labels
        eDepProfileX.SetTitle(eDepProfileX.GetTitle().replace("Energy deposition", "Charge deposited"))
        eDepProfileY.SetTitle(eDepProfileY.GetTitle().replace("Energy deposition", "Charge deposited"))
        eDepProfileX.SetName("Charge deposited [C]")
        eDepProfileY.SetName("Charge deposited [C]")
        
        eDepProfileX.SetDirectory(0), eDepProfileY.SetDirectory(0)
        return (eDepProfileX, eDepProfileY)


    # Loads the MC data from the fname, slice the total energy deposition in the sensor into bunches with 'bunchParNb' particles each.
    def getDepChgProfs(self, bunchParNb: int):
        """
        Loads the MC data from the fname, slice the total energy deposition in the sensor into bunches with 'bunchParNb' particles each,
        and returns a collection of TGraphErrors with the deposited energy profiles (cce=1 and avgPairEn=1 are used for this). Errors attached to them.
        The errors are calculated in the following way:
        1. the strip charge i is evaluated for each bunch number and the std is calculated,
        2. to the strip charge i of each bunch is then attached this std value.
        
        Parameters
        ----------
            bunchParNb (int) : number of particles (electrons or gamma) in a bunch
        """
        
        # Attempt to load from file
        dumpFname = (self.rFname).replace('.root', f'_bunchParNb{bunchParNb}.npy')    
        if os.path.exists(dumpFname):
            msg = f"Loading {dumpFname[dumpFname.rfind('/')+1:]} from file"
            try:
                self.bunches_chgDepProfiles = np.load(dumpFname, allow_pickle=True)
                msg += " with "+"\x1b[32;1m"+"success"+"\x1b[0m"
                (self.logging).debug(msg)
            except Exception as e:
                msg += f" failed ({e})"
                (self.logging).error(msg)
                (self.logging).debug("Calculating from scratch")
        

        # Get from the edepMaps bunch-by-bunch from the ROOT file produced by the MC
        # (by calling readEdepFromROOT) the bunches are sliced and the number of bunches is returned
        bunchesNb = self.readEdepFromROOT(bunchParNb)
        
        # Calculate the statistical error to be attached to each strip
        npChgDepProfile_err = self.calculateProfileStatErrs(self.bunches_edepMap)
        
        # Attach the errors to the profiles
        chgDepProfsBunch = []
        for bunch in range(bunchesNb):
            # Get the bunch edep 2D Map
            _edepMapUp, _edepMapDo = self.bunches_edepMap[bunch][f"b{bunch}_edepMapUp"], self.bunches_edepMap[bunch][f"b{bunch}_edepMapDo"]
            
            # Get the charge projected profile upstream downstream in both directions
            chgDepProfileX_up, chgDepProfileY_up = self.fromEnergyToChargeDeposited(_edepMapUp)
            chgDepProfileX_do, chgDepProfileY_do = self.fromEnergyToChargeDeposited(_edepMapDo)
            
            # Attach the errors to the bunch profiles
            for i in range(200):
                chgDepProfileX_up.SetBinError(i, (npChgDepProfile_err[0, 0, i]) )
                chgDepProfileY_do.SetBinError(i, (npChgDepProfile_err[1, 1, i]) )
            
            # Create a TGraph with errors and attach the profile of projected charge from bunch i
            bunchGraphs = { 'd0_x' : ROOT.TGraphErrors(chgDepProfileX_up), 'd1_y' : ROOT.TGraphErrors(chgDepProfileY_do) }
            bunchGraphs['d0_x'].SetName(f"{self.namespace}b{bunch}_chgDep");  bunchGraphs['d1_y'].SetName(f"{self.namespace}b{bunch}_chgDep")
            chgDepProfsBunch.append(bunchGraphs)
            
            # Clean the memory for the getChgDepProjProfiles objects
            chgDepProfileX_up.Delete(); chgDepProfileY_up.Delete(); chgDepProfileX_do.Delete(); chgDepProfileY_do.Delete()
                    
        
        # Dump the processed profiles in memory into a file
        chgDepProfsBunch_np = np.array(chgDepProfsBunch)
        np.save(dumpFname, chgDepProfsBunch_np, allow_pickle=True)
        (self.logging).debug(f"[splitBunchesFromROOT] File {dumpFname[dumpFname.rfind('/')+1:]} saved on file")
            
        # Store in internal variable the
        self.bunches_chgDepProfiles = chgDepProfsBunch_np

    
    #################################################################################################

    # Take the profiles of deposited charge and create the profiles of projected charge
    def makeProjChgProfs(self, cce: float):
        """
        Take the profiles of deposited charge and create the profiles of projected charge
        """
        
        bunches_chgProjProfiles = []
        for bID, bunch in enumerate(self.bunches_chgDepProfiles):
            # Take the profiles of deposited charge
            chgProjProf_d0_x = bunch['d0_x'].Clone()
            chgProjProf_d1_y = bunch['d1_y'].Clone()
            
            # Set the labels accordingle to what these are going to be
            chgProjProf_d0_x.SetName(f"{self.namespace}b{bID}_chgProj_d0x")
            chgProjProf_d1_y.SetName(f"{self.namespace}b{bID}_chgProj_d1y")
            chgProjProf_d0_x.SetTitle("Horizontal profile of projected charge upstream [C]")
            chgProjProf_d1_y.SetTitle("Vertical profile of projected charge downstream [C]")
            
            # Geometrically project the charge deposited into projected charge
            chgProjProf_d0_x.Scale(cce)
            chgProjProf_d1_y.Scale(cce)
            
            bunches_chgProjProfiles.append({'d0_x': chgProjProf_d0_x, 'd1_y': chgProjProf_d1_y})
        
        # Store the profiles in the internal class variables
        self.bunches_chgProjProfiles = np.array(bunches_chgProjProfiles)
            
        

    #################################################################################################

    def calcChgProjProfs(self) -> None:
        self.getDepChgProfs(self.bunchParNb)
        self.makeProjChgProfs(self.cce)


    def getChgProjProfs(self) -> np.ndarray:
        """
        Loads the MC data from the fname, slice the total energy deposition in the sensor into bunches with 'bunchParNb' particles each,
        and returns a collection of TGraphErrors with the projected charge profiles (cce and avgPairEn are used for this) whose strip charges have also
        errors attached to them. The errors are calculated in the following way:
        1. the strip charge i is evaluated for each bunch number and the std is calculated,
        2. to the strip charge i of each bunch is then attached this std value.
        
        Parameters
        ----------

        Returns
        -------
            depProjProfsBunch_np (np.array) : list with d0_x and d1_y TGraphErrors with the projected charge profiles
        """

        return self.bunches_chgProjProfiles



# Test function for the class
if __name__ == "__main__":
    rMCClass = [readFromMc("build/dummyRun_100k.root", 10000, cce, 54.0) for cce in np.linspace(0.1, 0.8, 10)]
    canvases = [ROOT.TCanvas(f"canvas{i}", f"canvas{i}", 800, 600) for i in range(len(rMCClass))]
    for i, item in enumerate(rMCClass):
        canvas = canvases[i]
        canvas.Divide(3)
        canvas.cd(1)
        prjX = item.bunches_edepMap[0]['b0_edepMapUp'].ProjectionX()
        prjX.Draw("hist")
        canvas.cd(2)
        item.bunches_chgDepProfiles[0]['d0_x'].Draw("AP")
        canvas.cd(3)
        item.bunches_chgProjProfiles[0]['d0_x'].Draw("AP")
        canvas.Update()
    input()
