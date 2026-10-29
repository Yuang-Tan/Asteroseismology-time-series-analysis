# Theory behind the time-series analysis

This note explains the lecture theory used to interpret the [main assignment](main_notes.ipynb). The notebooks contain the measurements and numerical comparisons; the [extra ideas](extra_ideas_notes.ipynb) also describe their particular CLEAN and filtering methods.

## 1. Harmonic fitting and spectral normalization

A trial frequency asks how well a sinusoid explains the observations. Both sine and cosine terms are required because the phase is unknown. Weighted least squares minimizes

$$
Q(f)=\sum_i w_i\left[y_i-\alpha\sin(\omega t_i)-\beta\cos(\omega t_i)\right]^2,
\qquad \omega=2\pi f_{\mu\mathrm{Hz}}10^{-6}.
$$

Times are in seconds relative to an arbitrary origin. Changing that origin rotates the sine/cosine coefficients but preserves the fitted amplitude. For uneven sampling, the sine and cosine columns generally are not orthogonal. The Frandsen prescription retains their weighted cross term in the normal equations; simply projecting onto each column separately need not give the least-squares solution [1]. The main note gives the explicit coefficients.

The project reports

$$
A=\sqrt{\alpha^2+\beta^2},\qquad P=A^2.
$$

An isolated noise-free sinusoid with a known baseline therefore recovers its injected amplitude at the correct frequency. Velocity power has units of squared metres per second. This is an amplitude-squared spectrum, not a power spectral density (PSD). A PSD has an additional inverse-frequency unit and a normalization relating its integral to variance. Summing this project's spectral samples does not automatically recover time-domain variance; overlapping responses and a denser frequency grid can change that sum.

For the main stellar calculations, the ordinary or weighted sample mean is removed before the two-column fit. This differs from fitting an intercept simultaneously at every frequency, especially when a low-frequency sinusoid correlates with a constant. The distinction is relevant when comparing the main estimator with the joint fits used in the extra ideas.

## 2. Cadence, aliases and observing duration

For strictly uniform sampling with interval Δt, the conventional Nyquist frequency is

$$
f_{\mathrm{Nyquist}}=\frac{1}{2\Delta t}.
$$

Sinusoids related by sampling-frequency aliases can produce the same discrete measurements [2]. Uneven observation times change these degeneracies. A median cadence alone does not define a universal Nyquist boundary for irregular data; the actual sampling pattern and spectral window must be examined.

Duration controls a different limitation. For a rectangular observing interval containing many cycles, the local response of an isolated coherent oscillation is approximately

$$
P(f_0+\delta f)\propto\operatorname{sinc}^2(\pi T\delta f),
\qquad \operatorname{sinc}(x)=\frac{\sin x}{x}.
$$

Its first zero lies at an offset of 1/T, and the power-peak full width at half maximum is approximately

$$
\mathrm{FWHM}_P\simeq\frac{0.8859}{T}.
$$

Longer observations narrow this response. A smaller frequency-grid step samples it more densely but does not create additional physical resolution. A strong signal can have a fitted frequency uncertainty smaller than the peak width; that local precision does not remove ambiguity between distant aliases. The synthetic duration experiment uses the nominal rectangular duration given by sample count times cadence.

## 3. Spectral windows and observing gaps

In Fourier analysis, observing a signal through a time-domain window means multiplying the signal by that window. The Fourier amplitudes are consequently convolved:

$$
x_{\mathrm{obs}}(t)=g(t)x(t),\qquad
\widetilde{x}_{\mathrm{obs}}(f)=\left(\widetilde{g}*\widetilde{x}\right)(f).
$$

This explains why a coherent oscillation can appear with a central peak and sidelobes. The identity concerns Fourier amplitudes; it is not a general convolution rule for the normalized, irregularly sampled least-squares power spectrum.

The assignment instead measures the estimator's sampling response directly. Unit sine and cosine reference signals are fitted at the actual timestamps, using the same weights as the observations, and their powers are averaged:

$$
W(f)=\frac{P_{\sin}(f)+P_{\cos}(f)}{2}.
$$

The phase average removes the arbitrary choice between these reference quadratures. Keeping their known zero baselines gives unit response at the reference frequency.

For two equal rectangular segments of length D separated by an end-to-start gap G, a local continuous-sampling approximation is [3]

$$
W(f_0+\delta f)\simeq
\operatorname{sinc}^2(\pi D\delta f)\cos^2\!\left[\pi(D+G)\delta f\right].
$$

The sinc envelope reflects each segment's length; the cosine modulation comes from interference between the separated intervals. Increasing the gap narrows the central feature while bringing strong competing features closer together. Different cycle counts across the unobserved interval can explain the measurements almost equally well. This is why a longer total baseline and a narrower central peak need not give an unambiguous frequency.

## 4. Statistical weights and noise

For independent measurements with known error standard deviations, inverse-variance weighting is appropriate:

$$
w_i\propto\frac{1}{\sigma_i^2}.
$$

Multiplying every weight by the same positive constant leaves the fitted coefficients unchanged. Relative weights can therefore improve the fit without establishing an absolute error scale. In this project, Procyon's third column already contains weights, whereas the 46 LMi scatter values must be inverse-squared. Whether the latter also describe absolute velocity errors is a separate question tested in Extra 5.

Weights alter the effective temporal coverage as well as the influence of noisy points [4]. Giving a few observing intervals most of the weight may lower the measured spectral background while broadening the main window response or strengthening daily aliases. Both the spectrum and the corresponding weighted window are needed to assess the trade-off.

White noise has a flat underlying PSD; correlated noise can have a frequency-dependent background. A high-frequency background estimate need not describe the noise near every stellar mode. Mean spectral amplitude is also distinct from time-domain RMS. These distinctions limit how the Procyon background reduction should be interpreted.

## 5. Coherence, damping and mode linewidths

A coherent oscillator retains a predictable phase. A stochastically excited, damped mode instead varies in amplitude and phase. In the ideal stationary, resolved case, its ensemble-mean spectral profile is Lorentzian [4]:

$$
\langle P(f)\rangle\propto
\frac{1}{1+4(f-f_0)^2/\Gamma^2},
\qquad \Gamma=\frac{1}{\pi\tau}.
$$

Here Γ is the full width at half maximum in cyclic frequency and τ is the amplitude-decay time. The relation does not use an energy-decay time or an angular-frequency linewidth.

A long-lived unresolved mode can resemble the finite observing-window response. A shorter-lived mode can be intrinsically broader. The observed profile also contains the effects of sampling, noise, neighbouring modes and possible rotational splitting. A single realization need not look like a smooth Lorentzian, so an observed peak width alone does not provide a reliable lifetime.

The solar comparison in the main assignment asks whether a coherent reference observed at identical times reproduces the local peak structure. Excess wing power indicates that this simple reference is insufficient. It supports considering a more complex signal model but does not by itself identify the physical cause, measure a mode lifetime or establish a detection significance.

## Lecture references

Page numbers below refer to pages of the supplied PDF files.

1. D. Stello, *PHYS4143 Time Series Lecture 1* (2026), pp. 38–40 and 61–64: spectral measurements, frequency resolution and harmonic fitting. The finite-sampling correction follows Frandsen et al. (1995), *A&A* 301, 123, Section 4.2, as reproduced in the lecture.
2. *Lecture 2*, pp. 18–40: sampling, Nyquist frequency and aliases.
3. *Lecture 3*, pp. 34–37 and 48–50: observing gaps, spectral windows and the two-segment approximation.
4. *Lecture 4*, pp. 9–23 and 24–33: statistical weights, noise, damping and Lorentzian profiles.
