#################################################################################################
# @info Simple digitization for the LUXE GP                                                     #
# @date 23/10/26                                                                                #
#                                                                                               #
# This script provides a simple way to map the energy deposited in the GBP sensor to a          #
# statistics of digitized profiles where the charge transport, propagamtion, digitization and   #
# front-end effects are accounted for.                                                          #
# TRACE and VERBOSE levels used for very detailed debugging                                     #
#################################################################################################
from multipledispatch import dispatch
from matplotlib import pyplot as plt
from sys import argv as CLIargs
from inspect import signature
from tqdm import tqdm
import numpy as np
import ROOT
import os

from .readFromMC import readFromMc
from .frontend import frontend
from .featureExtractor import featureExtractor

# digiManager.py logger
from .logger import create_logger
logging = create_logger("digiManager")
    
# Enable ROOT implicit multithreading
ROOT.EnableImplicitMT(16)


"""
TODO
1. [OK] Integrate the rdataStruct_OPT class
2. [TODO] Memory management for ROOT objects to cleanup useless stuffs
3. [CANCELED] (fast enough) Multithreading for parallel processing of the parameter space for the readMC and frontend classes
4. [OK] Setup the jobs for the calculation
5. [OK] Write down a class which calculated on the digitized profiles the parameters that we want to extract from the digitization pipeline
"""



# Apply the full pipeline from the incident radiation (readFromMc) to the digitized signal (ADC)
@dispatch(str, str, int, float, float, float, int, float, float, list)
def pipeline(mcPath: str, outPath: str, _bunchParNb: int, _cce: float, _avgPairEn: float, _fNoise: float, _iADCres: int, _fOlScale: float, _fGain: float, _vChgShrCrossTalkMap: list):
    """
    Apply the full pipeline from the incident radiation (readFromMc) to the digitized signal (ADC)
    
    Parameters
    ----------
        mcPath (str) : Filename of the MC ROOT file containing the data
        outPath (str) : Filename of the output ROOT file where the parameters from the feature extraction are stored
        _bunchParNb (int) : Number of particles in a bunch
        _cce (float) : Charge collection efficiency of the sensor for the bare geometrical projection of the dep chg. to proj chg.
        _avgPairEn (float) : Average energy required to create an electron-hole pair in the sensor (in eV)
        _fNoise (float) : Front-end electronics noise (in electron units) and internally converted in Coulomb
        _iADCres (int) : Number of bits of the analog-to-digital converter (12-bit -> 12)
        _fOlScale (float) : Full scale range that the frontend is capable of digitizing (in electron units)
        _fGain (float) : Amplifier gain (ration between final charge and initial charge)
        _vChgShrCrossTalkMap (list) : Cross-talk charge sharing percentages (es. [0.1,0.002, 0.0003] means that 0.1 is shared between strips at 1 distance, 0.002 between strips at distance 2, etc.)
        
    Returns
    -------
        None     
    """
    
    # Create the pipeline parameters
    _pipelinePars = {
        "bunchParNb" : _bunchParNb,
        "cce" : _cce,
        "avgPairEn" : _avgPairEn,
        "fNoise" : _fNoise,
        "iADCres" : _iADCres,
        "fOlScale" : _fOlScale,
        "fGain" : _fGain,
        "vChgShrCrossTalkMap" : np.array(_vChgShrCrossTalkMap + [0]*(10-len(_vChgShrCrossTalkMap)))
    }
    
    # Read data from the MC
    rMCClass = readFromMc(rFname = mcPath, bunchParNb = _bunchParNb, cce = _cce, avgPairEn = _avgPairEn)
    projChgProfs = rMCClass.getChgProjProfs()

    # Define the front-end settings
    #aFrontEnd = frontend(fNoise = (1e-15/1.60e-19), iADCres = 13, fOlScale = (12e-15/1.60e-19), fGain = 1.0, vChgShrCrossTalkMap = [0], projChgProfiles = tmp)
    aFrontEnd = frontend(projChgProfiles = projChgProfs, fNoise = _fNoise, iADCres = _iADCres, fOlScale = _fOlScale, fGain = _fGain, vChgShrCrossTalkMap = _vChgShrCrossTalkMap)
    aFrontEnd.doPipeline()
    
    # The following class calculated the observables that one is interested to extract from the profiles
    fExInstance = featureExtractor(roFname = outPath, dgtChgProfiles = aFrontEnd.projChgProfiles, pipelinePars = _pipelinePars)
    fExInstance.fitSchemeA()
    
    fExInstance.writeFeatures()
    logging.debug("Done")


# Overload of the pipeline function with default parameters
@dispatch()
def pipeline():
    """
    Overload of the pipeline function with default parameters
    
    Parameters
    ----------
    1. mcPath = "build/dummyRun_100k.root",
    2. outPath = "testFeatureExtractor.root"
    3. _bunchParNb = 10000,
    4. _cce = 0.2,
    5. _avgPairEn = 27.0,
    6. _fNoise = 0,
    7. _iADCres = 8, 
    8. _fOlScale = (10e-15/1.60e-19),
    9. _fGain = 1.0,
    10. _vChgShrCrossTalkMap = [0]
    """
    
    pipeline(
        "build/dummyRun_100k.root",
        "testFeatureExtractor.root",
        10000,
        0.2,
        27.0,
        0.0,
        8, 
        (100e-15/1.60e-19),
        1.0,
        [0.1,0.01,0.001])






# Generate a parameter space. The idea of this function is to optimize the phase space w.r.t. the naive linspace tensor product
@dispatch(tuple, tuple, tuple, tuple, tuple, tuple, tuple)
def makePhaseSpace(cce: tuple, avgPairEn: tuple, fNoise: tuple, iADCres: tuple, fOlScale: tuple, fGain: tuple, vChgShrCrossTalkMap: tuple) -> np.array:
    """
    Expand the input tuples and generate the phase space to sample with the simulation
    
    Parameters
    ----------
        cce (tuple) : Range for the cce in the form (cceStart, cceEnd, Nppoints). (It includes the endpoint)
        avgPairEn (tuple) : Range for the avgPairEn in the form (avgPairEnStart, avgPairEnEnd, Nppoints). (It includes the endpoint)
        fNoise (tuple) : Range for the fNoise in the form (fNoiseStart, fNoiseEnd, Nppoints). (It includes the endpoint)
        iADCres (tuple) : Range for the iADCres in the form (iADCresStart, iADCresEnd, Nppoints). (It includes the endpoint)
        fOlScale (tuple) : Range for the fOlScale in the form (fOlScaleStart, fOlScaleEnd, Nppoints). (It includes the endpoint)
        fGain (tuple) : Range for the fGain in the form (fGainStart, fGainEnd, Nppoints). (It includes the endpoint)
        vChgShrCrossTalkMap (tuple) : Range for the vChgShrCrossTalkMap in the form (vChgShrCrossTalkMapStart, vChgShrCrossTalkMapEnd, Nppoints). (It includes the endpoint)
    
    Returns
    -------
        phaseSpace (np.array) : stack with the phase space linstack
    """
    
    # Expand the input tuples
    cceA, cceB, cceN = cce
    avgPairEnA, avgPairEnB, avgPairEnN = avgPairEn
    fNoiseA, fNoiseB, fNoiseN = fNoise
    iADCresA, iADCresB, iADCresN = iADCres
    fOlScaleA, fOlScaleB, fOlScaleN = fOlScale
    fGainA, fGainB, fGainN = fGain
    vChgShrCrossTalkMapA, vChgShrCrossTalkMapB, vChgShrCrossTalkMapN = vChgShrCrossTalkMap
    
    # Throw exception if the parameters are unallowed
    if len(vChgShrCrossTalkMapA)!=len(vChgShrCrossTalkMapB):
        logging.critical(f"The length of vChgShrCrossTalkMapA and vChgShrCrossTalkMapB should be equal. Instead you have used {vChgShrCrossTalkMapA} and {vChgShrCrossTalkMapB} with length {len(vChgShrCrossTalkMapA)} and {len(vChgShrCrossTalkMapB)}, respectively.")
        raise Exception("Invalid vChgShrCrossTalkMapA or vChgShrCrossTalkMapB (len).")
    
    # Throw some warnings if the picked phase space is idiotic
    
    # Generate the linear spaces in a sensible way
    phaseSpace_cce = np.linspace(cceA, cceB, cceN, endpoint=True)
    phaseSpace_avgPairEn = np.linspace(avgPairEnA, avgPairEnB, avgPairEnN, endpoint=True)
    phaseSpace_fNoise = np.linspace(fNoiseA, fNoiseB, fNoiseN, endpoint=True)
    phaseSpace_iADCres = np.linspace(iADCresA, iADCresB, iADCresN, endpoint=True)
    phaseSpace_fOlScale = np.linspace(fOlScaleA, fOlScaleB, fOlScaleN, endpoint=True)
    phaseSpace_fGain = np.linspace(fGainA, fGainB, fGainN, endpoint=True)
    phaseSpace_vChgShrCrossTalkMap = np.linspace(vChgShrCrossTalkMapA, vChgShrCrossTalkMapB, vChgShrCrossTalkMapN, endpoint=True)
    
    phaseSpace = np.array((phaseSpace_cce, phaseSpace_avgPairEn, phaseSpace_fNoise, phaseSpace_iADCres, phaseSpace_fOlScale, phaseSpace_fGain, phaseSpace_vChgShrCrossTalkMap), dtype=object)    
    # Check PS consistency, for example throw exceptions on repetitions
    for item in phaseSpace:
        if np.unique(item) != item: logging.warning(f"The {item} contains repetitions! Please fix.")
    
    return phaseSpace


# Overload of makePhaseSpace generating a sensible phase space for the scan for LUXE GP with A5202 cards
@dispatch()
def makePhaseSpace() -> np.ndarray:
    """
    Generate a sensible phase space for the scan for LUXE GP suing A5202 cards
    """
    
    # Define the exponential decay law for the charge sharing among N-nearest neightbours
    def chgCrossTalkExpLaw(vChgShrCrossTalkNN_percent: float, plScale_um: float):
        """
        Assuming an exponential decay law for decrease of the charge sharing among N-nearest neightbours, it generates a list of chgSharing percentages
        staring from 'vChgShrCrossTalkNN_percent' for the closest strip and using the 'plScale_um' as the exponential decay scale.
        
        Parameters
        ----------
            vChgShrCrossTalkNN_percent (float) : charge sharing percentage for the nearest neighbour
            plScale_um (float) : exponential decay scale in um
        
        Returns
        -------
            vChgShrCrossTalkA (np.array) : array of charge sharing percentages
        """
        
        # Check that the plScale is positive
        if plScale_um < 0: raise Exception("plScale should be positive")
        
        # Generate the list itself
        distances = np.linspace(0.100, 10*0.100, 10, endpoint=True)
        vChgShrCrossTalkA = vChgShrCrossTalkNN_percent * np.exp(-distances/plScale_um)
        logging.debug(vChgShrCrossTalkA)
        
        # Check that the crosstalk array is valid and won't trigger the frontend class exception
        if 2*np.sum(vChgShrCrossTalkA) > 1.0:
            logging.fatal(f"Phase space var invalid for chgCrossTalkExpLaw with {vChgShrCrossTalkA}")
            raise Exception("Invalid chgCrossTalkExpLaw arguments")

        return vChgShrCrossTalkA
    
    
    # Generate the linear spaces in a sensible way
    phaseSpace_cce = np.linspace(0.05, 0.2, 2, endpoint=True)
    phaseSpace_avgPairEn = np.linspace(27.0, 27.0, 1, endpoint=True)
    phaseSpace_fNoise = np.linspace((14.615e-15/1.60e-19), (8.289e-15/1.60e-19), 10, endpoint=True)
    phaseSpace_iADCres = np.linspace(13, 13, 1, dtype=int, endpoint=True)   # the dtype casting to int is important to enforce that ADC values are integers
    phaseSpace_fOlScale = np.linspace((15.75e-12/1.60e-19), (1.79e-12/1.60e-19), 4, endpoint=True)
    phaseSpace_fGain = np.linspace(5, 63, 4, endpoint=True)
    phaseSpace_vChgShrCrossTalkMap = np.linspace(chgCrossTalkExpLaw(0.02, 0.100), chgCrossTalkExpLaw(0.1, 0.100), 2, endpoint=True)
    phaseSpace = np.array((phaseSpace_cce, phaseSpace_avgPairEn, phaseSpace_fNoise, phaseSpace_iADCres, phaseSpace_fOlScale, phaseSpace_fGain, phaseSpace_vChgShrCrossTalkMap), dtype=object)

    # Check PS consistency, for example throw exceptions on repetitions
    for item in phaseSpace:
        if len(np.unique(item))!=len(item): logging.warning(f"The {item} contains repetitions! Please fix.")
    
    return phaseSpace



# Overload of makePhaseSpace defaultuing the parameters (test method)
def makePhaseSpace_test() -> np.ndarray:
    """
    # Test method
    """
    
    # Generate the linear spaces in a sensible way
    phaseSpace_cce = np.linspace(0.2, 0.2, 1, endpoint=True)
    phaseSpace_avgPairEn = np.linspace(27.0, 30.0, 2, endpoint=True)
    phaseSpace_fNoise = np.linspace(0, 1000, 2, endpoint=True)
    phaseSpace_iADCres = np.linspace(13, 13, 1, dtype=np.intc, endpoint=True)   # the dtype casting to int is important to enforce that ADC values are integers
    phaseSpace_fOlScale = np.linspace((100e-15/1.60e-19), (1000e-15/1.60e-19), 2, endpoint=True)
    phaseSpace_fGain = np.linspace(1, 10, 2, endpoint=True)
    phaseSpace_vChgShrCrossTalkMap = np.linspace([0.010, 0.001], [0.10, 0.001], 2, endpoint=True)
    
    phaseSpace = np.array((phaseSpace_cce, phaseSpace_avgPairEn, phaseSpace_fNoise, phaseSpace_iADCres, phaseSpace_fOlScale, phaseSpace_fGain, phaseSpace_vChgShrCrossTalkMap), dtype=object)
    
    # Check PS consistency, for example throw exceptions on repetitions
    for item in phaseSpace:
        if len(np.unique(item))!=len(item): logging.warning(f"The {item} contains repetitions! Please fix.")
    
    return phaseSpace






# Take an input phase space and perform the scan in the parameter space
@dispatch(str, str, int, np.ndarray)
def makeJobs(mcPath: str, dataPath: str, bunchParNb: int, phaseSpace: np.ndarray):
    """
    Take an input phase space and perform the scan in the parameter space
    
    Parameters
    ----------
        mcPath (str) : Filename of the MC ROOT file containing the data
        dataPath (str) : Path where the ROOT files produced by the script are saved
        bunchParNb (int) : Number of particles in a bunch
        phaseSpace (np.array) : Phase space where the scan is performed
    
    Pipeline Pars
    ----------
        outPath (str) : Filename of the output ROOT file where the parameters from the feature extraction are stored
        _cce (float) : Charge collection efficiency of the sensor for the bare geometrical projection of the dep chg. to proj chg.
        _avgPairEn (float) : Average energy required to create an electron-hole pair in the sensor (in eV)
        _fNoise (float) : Front-end electronics noise (in electron units) and internally converted in Coulomb
        _iADCres (int) : Number of bits of the analog-to-digital converter (12-bit -> 12)
        _fOlScale (float) : Full scale range that the frontend is capable of digitizing (in electron units)
        _fGain (float) : Amplifier gain (ration between final charge and initial charge)
        _vChgShrCrossTalkMap (list) : Cross-talk charge sharing percentages (es. [0.1, 0.002, 0.0003] means that 0.1 is shared between strips at 1 distance, 0.002 between strips at distance 2, etc.)
        
    Returns
    -------
        None     
    """
    
    # Make sure the data path has the final '/' 
    if len(dataPath) > 0 and dataPath[-1] != '/': dataPath += '/'
    
    # Create the filename of the output file in such a way that it is intellegible what pars defined the run
    def createFname(_bunchParNb, _cce, _avgPairEn, _fNoise, _iADCres, _fOlScale, _fGain, _vChgShrCrossTalkMap):
        """
        Build a string uniquely identifying the file it is going to be generated in the function
        
        Returns
        -------
            ofFname (str) : the output filename
        """
        _vChgShrCrossTalkMapStr = ""
        for item in _vChgShrCrossTalkMap:
            _vChgShrCrossTalkMapStr += f"{item:.3e},"
        if len(_vChgShrCrossTalkMap): _vChgShrCrossTalkMapStr = _vChgShrCrossTalkMapStr[:-1]
        ofFname = f"mkJbs_{_bunchParNb}_{_cce:.2f}_{_avgPairEn:.1f}_{_fNoise:.0f}_{_iADCres}_{_fOlScale:.0f}_{_fGain:.3f}_{_vChgShrCrossTalkMapStr}.root"
        return dataPath+ofFname
    
    # Colours because life is poor without them
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;1m"
    grey = "\x1b[38;20m"
    red = "\x1b[31;20m"
    yellow = "\x1b[33;20m"
    reset = "\x1b[0m"
    
    # Nested loop for the scanning of the phase space
    for cce in tqdm(phaseSpace[0],  desc=bold_red+"cce"+reset, position=0):
        for avgPairEn in tqdm(phaseSpace[1], desc=green+" avgPE"+reset, position=1, leave=False):
            for fNoise in tqdm(phaseSpace[2], desc=red+"  fNois"+reset, position=2, leave=False):
                for iADCres in tqdm(phaseSpace[3], desc="   iADCr", position=3, leave=False):
                    for fOlScale in tqdm(phaseSpace[4], desc=yellow+"    fOlSc"+reset, position=4, leave=False):
                        for fGain in tqdm(phaseSpace[5], desc=grey+"     fGain"+reset, position=5, leave=False):
                            for vChgShrCrossTalkMap in tqdm(phaseSpace[6], desc=green+"      vChgShrCrossTalkMap"+reset, position=6, leave=False):
                                #print(mcPath, createFname(bunchParNb, cce, avgPairEn, fNoise, iADCres, fOlScale, fGain, vChgShrCrossTalkMap), bunchParNb, cce, avgPairEn, fNoise, iADCres, fOlScale, fGain, list(vChgShrCrossTalkMap))
                                #continue
                                roFname = createFname(bunchParNb, cce, avgPairEn, fNoise, iADCres, fOlScale, fGain, vChgShrCrossTalkMap)
                                if os.path.exists(roFname): continue
                                pipeline(mcPath, roFname, bunchParNb, float(cce), float(avgPairEn), float(fNoise), int(iADCres), float(fOlScale), float(fGain), list(vChgShrCrossTalkMap))


# Overload defaulting the path to data
@dispatch(str, int, np.ndarray)
def makeJobs(mcPath: str, bunchParNb: int, phaseSpace: np.ndarray):
    return makeJobs(mcPath, "data/", bunchParNb, phaseSpace)


# Overload of the makeJobs for CLI usage
@dispatch(str, str, int, float, float, int, float, float, int, float, float, int, int, int, int, float, float, int, float, float, int, list, list, int)
def makeJobs(mcPath: str, dataPath: str, bunchParNb: int, cceA: float, cceB: float, cceN: int, avgPairEnA: float, avgPairEnB: float, avgPairEnN: int, fNoiseA: float, fNoiseB: float, fNoiseN: int, iADCresA: int, iADCresB: int, iADCresN: int, fOlScaleA: float, fOlScaleB: float, fOlScaleN: int, fGainA: float, fGainB: float, fGainN: int, vChgShrCrossTalkMapA: list, vChgShrCrossTalkMapB: list, vChgShrCrossTalkMapN: int):    
    """
    Overload of the makeJobs function to accept input parameters from the CLI
    
    Parameters
    ----------
        mcPath (str) : Filename of the MC ROOT file containing the data
        dataPath (str) : Path where the ROOT files produced by the script are saved
        bunchParNb (int) : Number of particles in a bunch
        cce (tuple) : Range for the cce in the form (cceStart, cceEnd, Nppoints). (It includes the endpoint)
        avgPairEn (tuple) : Range for the avgPairEn in the form (avgPairEnStart, avgPairEnEnd, Nppoints). (It includes the endpoint)
        fNoise (tuple) : Range for the fNoise in the form (fNoiseStart, fNoiseEnd, Nppoints). (It includes the endpoint)
        iADCres (tuple) : Range for the iADCres in the form (iADCresStart, iADCresEnd, Nppoints). (It includes the endpoint)
        fOlScale (tuple) : Range for the fOlScale in the form (fOlScaleStart, fOlScaleEnd, Nppoints). (It includes the endpoint)
        fGain (tuple) : Range for the fGain in the form (fGainStart, fGainEnd, Nppoints). (It includes the endpoint)
        vChgShrCrossTalkMap (tuple) : Range for the vChgShrCrossTalkMap in the form (vChgShrCrossTalkMapStart, vChgShrCrossTalkMapEnd, Nppoints). (It includes the endpoint)
                
    Returns
    -------
        None   
    """

    # Validate input parameters
    ## Make sure that the file in the mcPath exists    
    if not os.path.exists(mcPath): raise Exception(f"File {mcPath} does not exist")
    if bunchParNb <= 0: raise Exception(f"bunchParNb should be positive")

    try:
        # Fix the vChgShrCrossTalkMapA, vChgShrCrossTalkMapB instances
        vChgShrCrossTalkMapA = [float(entry) for entry in (''.join(vChgShrCrossTalkMapA))[1:-1].split(',')]
        vChgShrCrossTalkMapB = [float(entry) for entry in (''.join(vChgShrCrossTalkMapB))[1:-1].split(',')]
    except:
        pass
        
    # Create the phase space with the default PS function generator
    pS = makePhaseSpace(
        (cceA, cceB, cceN),
        (avgPairEnA, avgPairEnB, avgPairEnN),
        (fNoiseA, fNoiseB, fNoiseN),
        (iADCresA, iADCresB, iADCresN),
        (fOlScaleA, fOlScaleB, fOlScaleN),
        (fGainA, fGainB, fGainN),
        (vChgShrCrossTalkMapA, vChgShrCrossTalkMapB, vChgShrCrossTalkMapN))


    # Run the pipeline scan
    makeJobs(mcPath, dataPath, bunchParNb, pS)






if __name__=="__main__":
    pass
    #logging.setLevel(10)       # This correspond to debug mode
    
    # Example running the pipeline of digitization
    pipeline()
    exit()

    # Example running a digitization pipeline scanning a dummy phasespace
    pS = makePhaseSpace_test()
    dataDir, bunchParNb = "data/", 10000
    makeJobs("build/dummyRun_100k.root", dataDir, bunchParNb, pS)
    exit()
    
    # Example with a phasespace on A5202 physical grounds 
    #pS = makePhaseSpaceGPA5202()
    #dataDir, bunchParNb = "data/", 10000
    #makeJobs("build/dummyRun_100k.root", dataDir, bunchParNb, pS)
    #exit()


    





########################################################################################################################
########################################################################################################################
# Documentation
def printHelp():
    """
    Print the CLI documentation
    """
    # Take a copy of the list of symbols defined in the global scope
    globalSymbols = dict(globals())
    
    print("Usage: python digiManager.py <function_name> <arg1> <arg2> ...")
    print("Available functions are: ")
    count = 1
    for symbolName in globalSymbols:
        if symbolName == "printHelp": continue
        # Check if the symbol is a callable
        if callable(globals()[symbolName]):
            if count > 4: print(f"\t{count-4}. {symbolName}")
            count += 1

##############################
# Handle CLI arguments
if len(CLIargs) < 2:
    # Documentation
    printHelp()
    exit(0)
else:
    # Take the list of symbols defined in the global scope
    globalSymbols = globals()
    
    # Parse the first argument
    callableName = CLIargs[1]
    if (callableName not in globalSymbols):
        logging.error(f"Symbol {callableName} not found in the global scope")
        printHelp()
        exit(-1)
    elif not callable(globals()[callableName]):
        logging.error(f"Symbol {callableName} in globals but not a callable")
        printHelp()
        exit(-1)
    
    
    # Get the function object
    function_to_call = globals()[callableName]
    
    def get_signatures(dispatched_function):
        """
        Get the list of signatures for the dispatched function. Generalization of the signature function for the multidispatch
        """
        # Get the Signature of the callable and the input parameters
        function_signature = signature(dispatched_function)
        
        if (f"{function_signature}" == "(*args, **kwargs)"):
            # dispatch
            signatures = [entry[0] for entry in dispatched_function.funcs.items()]
        else:
            signatures = [entry[1].annotation for entry in function_signature.parameters.items()]
            signatures = list([tuple(signatures)])
        return signatures
    
    
    # Get the Signature of the callable and the input parameters and extract the names and types of parameters
    param_info = get_signatures(function_to_call)
    
    # Extract the function arguments from the command line
    args_from_cli = CLIargs[2:]
    
    # Check if the number of arguments matches the number of parameters
    argNbs_NotMatching = True
    for entry in param_info:
        if len(entry) == len(args_from_cli):
            argNbs_NotMatching = False
            break
    if argNbs_NotMatching:
        logging.error("Number of arguments does not match the function signature.")
        print(param_info)
        exit(-1)


    # Attempt to convert arguments to the expected types
    ## Generalization for multipledispatch
    argConvertedFail = True
    for par_types in param_info:
        toCastParNb = 0
        converted_args = []
        for argI, par_type in enumerate(par_types):
            #logging.debug(par_type, argI, args_from_cli[argI])
            #print(par_type, argI, args_from_cli[argI], end="\t")            
            try:
                if args_from_cli[argI].isdecimal() and par_type is str:
                    continue
                #if (args_from_cli[argI].replace('.', '').isdecimal()) and par_type in [float, np.double, np.single]:
                #    continue
                castedPar = par_type(args_from_cli[argI])
                converted_args.append(castedPar)
                toCastParNb += 1
                #print()
            except:
                #print("except")
                #print(f"Failed to convert {args_from_cli[argI]} to {par_type}. breaking to next set of args")
                break
        if len(converted_args) == len(args_from_cli):
            argConvertedFail = False
            break        
    if argConvertedFail:
        logging.error("Unable to convert the arguments to any of the expected types.")
        exit(-1)
        

    # Info message
    msg = f"Running: {callableName} with arguments "
    for arg in converted_args: msg += f"{arg} "
    msg = msg[:-1]
    logging.info(msg)
    
    # Call the function with the converted arguments
    function_to_call(*converted_args)
    
    # Goodbye
    msg = ["Whatever happens, happens. - Spike Spiegel", "Everything has a beginning and an end. - Jet Black" , "They say hunger is the best spice. - Spike Spiegel", "That's what she said. - M. Scott"]
    from random import choice
    logging.status(choice(msg)+". Goodbye :) \n")

########################################################################################################################
########################################################################################################################
