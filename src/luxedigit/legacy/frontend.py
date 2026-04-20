#################################################################################################
# @info Frontend class simulating the response of the electronics to the profile of charge at   #
#       the strips                                                                              #
# @date 23/10/26                                                                                #
#                                                                                               #
#################################################################################################
from multipledispatch import dispatch
from matplotlib import pyplot as plt
from .logger import create_logger
import numpy as np
import ROOT




class frontend:
    """
    # Description
    Class simulating the effect of the frontend electronics to the profile of projected charge at the strips. There are several components of this class, which can be configured by the user:
    1. Input noise (fNoise parameter in e unit). Adds random gaussian noise to the input profile of charge. It simulated some electronics noise before the pre-amplification stage takes place.
    2. Amplification, controlled by the fGain parameters (units of mV/pC). In this context, the pre-amplification module is effectively simulating the charge collection at the input capacitor and the actual amplification from the OpAmp amplifier.
    3. 
    
    ## Parameters
    ----------
        fNoise (float) : Front-end electronics noise (in electron units) and internally converted in Coulomb
        iADCres (int) : Number of bits of the analog-to-digital converter (12-bit -> 12)
        fOlScale (float) : Full scale range that the frontend is capable of digitizing (in electron units)
        fGain (float) : Amplifier gain (ration between final charge and initial charge)
        vChgShrCrossTalkMap (list) : Cross-talk charge sharing percentages (es. [0.1,0.002, 0.0003] means that 0.1 is shared between strips at 1 distance, 0.002 between strips at distance 2, etc.)
    """
    
    # frontend class logger
    logging = create_logger("frontend")
    
    
    # Initialize the front-end class internal variables
    def initialize(self, fNoise: float, iADCres: int, fOlScale: float, fGain: float, vChgShrCrossTalkMap: list):
        """
        Initialize the front-end class internal variables
        
        Parameters
        ----------
            fNoise (float) : Front-end electronics noise (in electron units) and internally converted in Coulomb
            iADCres (int) : Number of bits of the analog-to-digital converter (12-bit -> 12)
            fOlScale (float) : Full scale range that the frontend is capable of digitizing (in electron units)
            fGain (float) : Amplifier gain (ration between final charge and initial charge)
            vChgShrCrossTalkMap (list) : Cross-talk charge sharing percentages (es. [0.1,0.002, 0.0003] means that 0.1 is shared between strips at 1 distance, 0.002 between strips at distance 2, etc.)
        
        Returns
        -------
            None
        """
        
        # Set internal variables of the class with the external parameters
        self.fNoise             = fNoise * (1.602176e-19)
        self.iADCres            = iADCres
        self.fOlScale           = fOlScale * (1.602176e-19)
        self.fGain              = fGain
        self.chgShrCrossTalkMap = vChgShrCrossTalkMap
        
        self.maxADC             = np.power(2, self.iADCres) - 1  
        #self.adcScale           = self.vref/self.maxADC #in units of V/count
        
        
        if 2*sum(vChgShrCrossTalkMap) > 1.0:
            (self.logging).warning("Sum of charge sharing fractions cannot exceed 0.5. Cross talk will be disabled")
            self.chgShrCrossTalkMap = [0]
        
        if fGain <= 0:
            (self.logging).warning(f"Negative gain value of {fGain}. Defaulting to 1.0")
            self.fGain = 1.0
            
        msg = f"""Frontend settings are:
        fNoise              : {fNoise} e or {self.fNoise:.2f} in C
        iADCres             : {self.iADCres}-bit
        fOlScale            : {fOlScale} e or {self.fOlScale:.2f} in C
        fGain               : {fGain} or {self.fGain:.2f}
        chgShrCrossTalkMap  : {self.chgShrCrossTalkMap} % \n----------------
        """
        (self.logging).debug(msg)
     
    
    def __init__(self, fNoise: float, iADCres: int, fOlScale: float, fGain: float, vChgShrCrossTalkMap: list, projChgProfiles: np.ndarray):
        """
        Overload of the init class function for taking an input projProfile profile and processing with default parameters
        
        Parameters
        ----------
            fNoise (float) : Front-end electronics noise (in electron units) and internally converted in Coulomb
            iADCres (int) : Number of bits of the analog-to-digital converter (12-bit -> 12)
            fOlScale (float) : Full scale range that the frontend is capable of digitizing (in electron units)
            fGain (float) : Amplifier gain (ration between final charge and initial charge)
            vChgShrCrossTalkMap (list) : Cross-talk charge sharing percentages (es. [0.1,0.002, 0.0003] means that 0.1 is shared between strips at 1 distance, 0.002 between strips at distance 2, etc.)
            projChgProfiles (np.array) : Array with the projected charge profiles for the hor./vert. bunches
            
        Returns
        -------
            None
        """
        
        # Copy the list of TGraphErrors internally so that the input objects are not modified
        self.projChgProfiles = np.array([{'d0_x' : item['d0_x'].Clone(), 'd1_y': item['d1_y'].Clone()} for item in projChgProfiles])
        (self.logging).debug("Called with external profiles")
        self.initialize(fNoise, iADCres, fOlScale, fGain, vChgShrCrossTalkMap)
    
    
    # Apply a gaussian smearing noise to the input with sigma fNoise to the input profile
    @dispatch(ROOT.TGraphErrors)
    def applyNoise(self, chgStripProfile: ROOT.TGraphErrors):
        """
        Apply a gaussian smearing noise to the input with sigma fNoise to the input profile
        
        Paramters
        ----------
            chgStripProfile (ROOT.TGraphErrors) : profile of charge collected at the strips
        """
        
        # Set the name, title and vertical axis title
        chgStripProfile.SetTitle(chgStripProfile.GetTitle().replace("Charge collected", "Charge collected (plus FE-noise)"))
        chgStripProfile.SetName(chgStripProfile.GetName().replace('Proj', 'ProjFE'))    #chgProjFEProfileY
        chgStripProfile.GetYaxis().SetTitle("charge collected with FE-noise [C]")
        #chgStripProfileCp.GetXaxis().CenterTitle()
        #chgStripProfileCp.GetYaxis().CenterTitle()

        for i in range(chgStripProfile.GetN()):
            noise = np.random.normal(0, self.fNoise)
            chgStripProfile.SetPointY(i, chgStripProfile.GetPointY(i) + noise)
            chgStripProfile.SetPointError(i, chgStripProfile.GetErrorX(i), np.sqrt(chgStripProfile.GetErrorY(i)*chgStripProfile.GetErrorY(i) + self.fNoise*self.fNoise) )

        ## Diagnostic plots
        #if plotting == 10:
        #    view, ax = plt.subplots()
        #    view.suptitle("Profile with noise applied")
        #    ax.set_xlabel("strip no.")
        #    ax.set_ylabel("signal+noise [C]")
        #    ax.plot(noiseVals, label=f"noise set value is {np.round(self.fNoise,1)} e.")
        #    view.savefig("applyNoise.pdf")
        #    plt.show()
        
    
    # Apply a gaussian smearing noise to the input with sigma fNoise to the array of hor./vert. bunch profiles
    @dispatch()
    def applyNoise(self):
        for entry in self.projChgProfiles:
            self.applyNoise(entry['d0_x'])
            self.applyNoise(entry['d1_y'])

    
    # Simulate the crosstalk effect, sharing the projected charge at strip ith with neighbouring strips
    @dispatch(ROOT.TGraphErrors)
    def simulateCrosstalk(self, chgProfile: ROOT.TGraphErrors):
        """
        Simulates the cross-talk between neighbour strips. The cross talk is simulated by adding a fraction of the charge of each strip to the nearest neighbours, with a percentage depending on the distance between strip i and strip j

        Parameters
        ----------
            chgProfile (ROOT.TGraphErrors) : input profile of projected charge
        """
        
        stripNb = chgProfile.GetN()
        
        npProfile = np.zeros((stripNb,2))
        for i in range(stripNb):
            # Get the charge and error on it
            stripChg = chgProfile.GetPointY(i)
            stripChg_err = chgProfile.GetErrorY(i)
            npProfile[i, 0] = stripChg
            npProfile[i, 1] = stripChg_err

        # Create arrays to store the crosstalk contributions and their errors
        crosstalk_contribs = np.zeros((stripNb, stripNb))
        crosstalk_contribs_err = np.zeros((stripNb, stripNb))

        # Calculate the crosstalk contributions
        for i in range(stripNb):
            if npProfile[i, 0] != 0:
                for j in range(len(self.chgShrCrossTalkMap)):
                    crosstalk_contribs[i, j] = npProfile[i, 0] * (self.chgShrCrossTalkMap)[j]
                    crosstalk_contribs_err[i, j] = npProfile[i, 1] * (self.chgShrCrossTalkMap)[j]

        # Update the charge profile
        updatedProfile = npProfile.copy()
        #initialChg = np.sum(updatedProfile, 0)
        for i in range(stripNb):
            if npProfile[i, 0] != 0:
                for j in range(stripNb):
                    # Update the charge on the left neighbor
                    if i-(j+1) >= 0:
                        updatedProfile[i-(j+1), 0] += crosstalk_contribs[i, j]
                        updatedProfile[i-(j+1), 1] = np.sqrt(updatedProfile[i-(j+1), 1]**2 + crosstalk_contribs_err[i, j]**2)
                    
                    # Update the charge on the right neighbor
                    if i+(j+1) < stripNb:
                        updatedProfile[i+(j+1), 0] += crosstalk_contribs[i, j]
                        updatedProfile[i+(j+1), 1] = np.sqrt(updatedProfile[i+(j+1), 1]**2 + crosstalk_contribs_err[i, j]**2)
                    
                    # Update the charge on current strip  
                    updatedProfile[i, 0] -= 2*crosstalk_contribs[i, j]
                    updatedProfile[i, 1] = 2*np.sqrt(updatedProfile[i, 1]**2 + crosstalk_contribs_err[i, j]**2)
                    
        ## Normalize the total charge shared
        #currentChg = np.sum(updatedProfile, 0)
        #updatedProfile *=  initialChg/currentChg
        
        for i in range(stripNb):
            chgProfile.SetPointY(i, updatedProfile[i, 0])

    
    # Overload of simulateCrosstalk for the array of hor./vert. bunch profiles
    @dispatch()
    def simulateCrosstalk(self):
        for entry in self.projChgProfiles:
            self.simulateCrosstalk(entry['d0_x'])
            self.simulateCrosstalk(entry['d1_y'])
    
    
    # Simulate the charge amplification, effectively by rescaling the input profile by the gain factor
    @dispatch(ROOT.TGraphErrors)
    def applyAmplification(self, chgStripWithFE: ROOT.TGraphErrors):
        """
        Simulate the charge amplification, effectively by rescaling the input profile by the gain factor
        
        Paramters
        ----------
            chgStripWithFE (ROOT.TGraphErrors) : profile of the charge projected at the strips with FE noise applied
            
        """
        
        ## Get sensor name, if upstream or downstream
        #sensor = "upstream"
        #if 'downstream' in chgStripWithFE.GetTitle(): sensor = "downstream"
        ## Set title
        #chgStripWithFE.SetTitle(f"Low gain ADC-input voltage {sensor} sensor")
        #hgADCinVoltProf.SetTitle(f"High gain ADC-input voltage {sensor} sensor")
        ## Set name (chgStripWithFE name is chgProjFEProfileY) is mapped into lgADCinVoltProfX or lgADCinVoltProfY
        #chgStripWithFE.SetName("lgADCinVoltProf"+(chgStripWithFE.GetName())[-1])
        #hgADCinVoltProf.SetName("hgADCinVoltProf"+(chgStripWithFE.GetName())[-1])
        ## Set vertical axis title
        #chgStripWithFE.GetYaxis().SetTitle("LG ADC-input (V)")
        #hgADCinVoltProf.GetYaxis().SetTitle("HG ADC-input (V)")
        #chgStripWithFE.GetXaxis().CenterTitle()
        #chgStripWithFE.GetYaxis().CenterTitle()
        #hgADCinVoltProf.GetXaxis().CenterTitle()
        #hgADCinVoltProf.GetYaxis().CenterTitle()
        
        # Convert the charge to a voltage (the histograms contains chg. in C, the pre-factor 1e9 converts the mV/pC to V/C)
        chgStripWithFE.Scale(self.fGain)

        ## Diagnostic plots
        #if ((self.logging).level == 10):
        #    view, ax = plt.subplots()
        #    view.suptitle("Profile with amplification applied")
        #    ax.set_xlabel("strip no.")
        #    ax.set_ylabel("signal+noise [V]")
        #    ax.plot(chgStripWithFE, label=f"low gain set value is {np.round(self.lGain,1)} mV/pC.")
        #    ax.plot(hgADCinVoltProf, label=f"high gain set value is {np.round(self.hGain,1)} mV/pC.")
        #    view.legend(loc="upper right")
        #    view.show()
        #    plt.show()
    
    
    # Overload for apply amplification for the projChgProfiles array 
    @dispatch()
    def applyAmplification(self):
        for entry in self.projChgProfiles:
            self.applyAmplification(entry['d0_x'])
            self.applyAmplification(entry['d1_y'])
    
    
     # Simulates the cross talk between two strips of the sensors. The cross talk is simulated by adding a normal distribution centered around the selected strip number for each sensor.
    

    # Simulate the ADC conversion by discretizing the input profile with a step function
    @dispatch(ROOT.TGraphErrors)
    def applyADC(self, chgStripAmp: ROOT.TGraphErrors):
        """
        Simulate the ADC conversion by discretizing the input profile with a step function
        
        Paramters
        ----------
            chgStripAmp (ROOT.TGraphErrors) : input profile
        """
        
        ## Debugging info about ADC module
        #msg = f"""Applying AD conversion with parameters:
        #adcResolution      : {self.iADCres}-bit
        #maxADC             : {self.maxADC}
        #adcScale           : {self.adcScale} V/count\n----------------"""
        #(self.logging).debug(msg)

        ## Set the proper names, styles etc.
        #chgStripAmp.SetTitle(chgStripAmp.GetTitle().replace("ADC-input voltage", "ADC-counts"))
        #chgStripAmp.GetYaxis().SetTitle(f"ADC counts [0-{self.maxADC}]")
        #chgStripAmp.SetName(chgStripAmp.GetName().replace('inVolt', 'cts'))
        
        for i in range(chgStripAmp.GetN()):
            # Get charge
            chg = chgStripAmp.GetPointY(i)
            chg_err = chgStripAmp.GetErrorY(i)
            
            # Discretization
            if chg < 0:
                adcCounts = 0
                adccounts_err = 0
            elif chg >= self.fOlScale:
                adcCounts = self.maxADC
                adccounts_err = 0
            else:
                adcCounts = np.intc((chg/self.fOlScale) * self.maxADC)
                adccounts_err = np.intc(chg_err * self.maxADC/self.fOlScale)
            
            # Set digitized charge
            chgStripAmp.SetPointY(i, adcCounts)
            chgStripAmp.SetPointError(i, 1.4433757e-05, adccounts_err)


        ## Diagnostic plots
        #if ((self.logging).level == 10):
        #    view, ax = plt.subplots()
        #    view.suptitle("Profile of ADC counts")
        #    ax.set_xlabel("strip no.")
        #    ax.set_ylabel("ADC counts")
        #    ax.plot(chgStripAmp, label=f"ADC resolution is {self.adcResolution}-bit.")
        #    view.legend(loc="upper right")
        #    view.show()
        #    plt.show()


    # Overload for apply amplification for the projChgProfiles array 
    @dispatch()
    def applyADC(self):
        for entry in self.projChgProfiles:
            self.applyADC(entry['d0_x'])
            self.applyADC(entry['d1_y'])
    
    
    # Apply the entire pipeline of the frontend class
    @dispatch(ROOT.TGraphErrors)
    def doPipeline(self, profile: ROOT.TGraphErrors):
        self.applyNoise(profile)
        self.simulateCrosstalk(profile)
        self.applyAmplification(profile)
        self.applyADC(profile)
   
   
    # Overload of doPipeline for the projChgProfiles array
    @dispatch()
    def doPipeline(self):
        for entry in self.projChgProfiles:
            self.doPipeline(entry['d0_x'])
            self.doPipeline(entry['d1_y'])
   
    
    # Return the array of digitized profiles
    def getDigitizedProfiles(self) -> np.ndarray:
        return (self.projChgProfiles)
