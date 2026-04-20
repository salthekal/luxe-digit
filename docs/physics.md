# Physics context — LUXE and the Gamma Beam Profiler

This document gives a brief physics-level orientation for readers less
familiar with the LUXE experiment. It is not a substitute for the
collaboration's technical design report — it is here so that a reviewer of
this code understands *why* a digitization pipeline matters at all.

---

## 1 · Why strong-field QED

Quantum electrodynamics in the perturbative regime is among the most
precisely tested theories in physics. What remains experimentally
underexplored is its behavior at field strengths approaching the **Schwinger
critical field**,

$$E_{\rm cr} = \frac{m_e^2 c^3}{e \hbar} \approx 1.32 \times 10^{18}\ \mathrm{V/m}.$$

At fields of this magnitude the QED vacuum is expected to become non-linear:
pair production from the vacuum, photon–photon scattering, and harmonic
generation all become measurable. Directly producing such fields in the lab
is impossible with current laser technology — but the fields seen by an
ultra-relativistic electron colliding head-on with a focused laser pulse can
easily approach $E_{\rm cr}$ in the electron's rest frame.

The dimensionless intensity parameter

$$\xi = \frac{eE_0}{m_e c\, \omega_L}$$

quantifies the transition. For $\xi \gtrsim 1$, perturbation theory in the
laser field breaks down and all orders must be resummed — the *non-linear*
regime. **LUXE** (*Laser Und XFEL Experiment*) is designed to reach
$\xi \sim 1$ and beyond.

## 2 · The LUXE experiment

LUXE is a proposed experiment at DESY that collides the
**16.5 GeV electron beam** from the European XFEL with a high-intensity
laser pulse of peak intensity up to $10^{20}\ \mathrm{W/cm^2}$. Two running
modes are planned:

1. **e-laser mode.** The electron beam interacts directly with the laser,
   producing non-linear Compton photons and, at higher laser intensity,
   trident pair production.
2. **γ-laser mode.** The electron beam first produces bremsstrahlung
   photons on a tungsten target; these photons then interact with the
   laser, probing non-linear Breit–Wheeler pair production — a process
   never directly observed.

Both modes require careful characterization of the photon flux downstream
of the interaction point. That is the job of the **Gamma Beam Profiler**.

## 3 · The Gamma Beam Profiler (GBP)

The GBP is a silicon-strip detector assembly placed downstream of the
interaction region. Its purpose is to measure the *transverse spatial
profile* of the Compton-scattered gamma flux. This profile constrains:

* The electron–laser spatial overlap at the interaction point.
* Beam-related and secondary backgrounds.
* The normalization of the physics signals measured by the photon
  spectrometer and pair calorimeters.

The sensor is read out by **CAEN DT5202** front-end boards — 64-channel
charge-integrating ADCs with programmable gain and configurable full-scale
range. The signal path is, schematically:

```
silicon strip → charge collection → pre-amp → shaper → ADC → digital readout
```

Each stage adds noise, crosstalk, saturation effects, or quantization
errors. These are precisely the effects the code in this repository
simulates.

## 4 · Why digitization fidelity matters

Physics observables — photon spectrum, pair-yield, laser-intensity
inference — are extracted from *fit parameters* of the digitized strip
profiles (Gaussian mean, width, amplitude, baseline). Every digitization
effect distorts these extracted parameters:

| Effect | Distortion mode |
|---|---|
| Front-end noise | Widens the apparent $\sigma$, biases baseline |
| Crosstalk | Broadens and redistributes the Gaussian; can bias mean if asymmetric |
| ADC quantization | Introduces dead bands and step functions in the response |
| Saturation (clipping to $F_{\rm OL}$) | Heavily biases amplitude; requires fit rejection |
| Gain mis-calibration | Scales amplitude → propagates to absolute flux measurement |

Without a Monte Carlo characterization of these distortions, the
propagation from digitized profile to physical observable is opaque — and
the systematic uncertainties on the physics results balloon. The
digitization pipeline lets us study each effect in isolation, scan through
hardware configurations before the detector is built, and ultimately invert
the digitization to reconstruct the *true* deposited-energy profile from
the measured digital data.

## 5 · Why a learned feature extractor

The upstream `featureExtractor` uses a ROOT `TF1` Gaussian-plus-constant
fit with rejection regions for saturated strips. This works — it is the
right choice for an authoritative reference — but has four limitations that
an ML regressor addresses:

1. **Speed.** Each Minuit fit takes roughly $10\ \mathrm{ms}$ on CPU.
   For the parameter-space scans that define the detector optimization, we
   evaluate $\mathcal{O}(10^6)$ profiles. The MLP runs inference in
   $\sim 0.1\ \mathrm{ms}/\mathrm{profile}$ — two orders of magnitude
   faster, enabling scans that are currently impractical.

2. **Saturation handling.** The upstream fit branches based on whether the
   profile is saturating, with different rejection logic for each case. A
   learned regressor trained on both regimes handles the transition without
   explicit branching, provided the training set covers the relevant phase
   space.

3. **Epistemic uncertainty.** The Minuit errors are *statistical* — they
   describe the fit's sensitivity to the data, assuming the model is
   correct. MC-dropout (Gal & Ghahramani, 2016) approximates the posterior
   over model weights, giving us a second, independent uncertainty
   channel. Where the two diverge is where our MC training distribution
   under-covers the real data — a useful diagnostic.

4. **Differentiability.** The MLP is end-to-end differentiable with respect
   to its inputs. This permits future extensions in which the upstream
   digitization parameters are fit *jointly* with the physics observables
   via gradient-based inference — currently impossible with the piecewise
   Minuit approach.

A learned extractor does not replace the reference fit — it complements it.
Where the two agree, we trust both. Where they disagree, something
interesting is happening, and the disagreement is itself informative.

---

## References

* LUXE Collaboration, *Conceptual Design Report*, arXiv:2102.02032 (2021).
* LUXE Collaboration, *Technical Design Report*, Eur. Phys. J. ST 230, 2445 (2021).
* Gal, Y. & Ghahramani, Z., *Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning*, ICML (2016).
  arXiv:1506.02142.
* Schwinger, J., *On gauge invariance and vacuum polarization*, Phys. Rev.
  82, 664 (1951).
