"""BlueberryCircus -- a classical stochastic-electrodynamics (SED) simulator.

Nanarch Technologies, Inc.

A library in the lineage of Strawberry Fields / Mr Mustard / PennyLane, but for
*classical* stochastic electrodynamics: the dynamics of a charged point particle
bound by a potential and immersed in a classical zero-point-fluctuation
(ZPF) electromagnetic background, whose electric energy is stored capacitively
through the vacuum permittivity ``u_E = (1/2) eps0 <E^2>`` (Puthoff 1987;
Boyer 1975; Cole & Zou 2003; Nieuwenhuizen & Liska 2015).
"""
from .constants import (Units, SI, BOHR, EPS0, HBAR, C, E_CHARGE, M_E, K_E,
                        ALPHA, A0, radiation_reaction_time,
                        setterfield_rescale)
from . import spectrum, oracles, observables, potentials, symplectic, rectification
from .spectrum import rho, spectral_density_Ex, mode_density, mode_energy
from .zpf import ZPFBackground
from .dynamics import Particle, Trajectory, integrate
from .certify import Certificate, RULES, PASS, FAIL, NULL, audit_overclaim, \
    save_bundle, load_bundle
from .program import (Program, Harmonic, Coulomb, ZPF, RadiationReaction,
                      Operation)
from .engine import Engine, Result

__version__ = "0.2.0"

__all__ = [
    "Units", "SI", "BOHR", "EPS0", "HBAR", "C", "E_CHARGE", "M_E", "K_E",
    "ALPHA", "A0", "radiation_reaction_time", "setterfield_rescale",
    "spectrum", "oracles", "observables", "potentials", "symplectic",
    "rectification", "rho", "spectral_density_Ex", "mode_density",
    "mode_energy",
    "ZPFBackground", "Particle", "Trajectory", "integrate", "Certificate",
    "RULES", "PASS", "FAIL", "NULL", "audit_overclaim", "save_bundle",
    "load_bundle", "Program", "Harmonic", "Coulomb", "ZPF", "RadiationReaction",
    "Operation", "Engine", "Result", "__version__",
]
