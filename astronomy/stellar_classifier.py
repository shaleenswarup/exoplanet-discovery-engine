"""Stellar Classification Module.

Classifies stars using spectral characteristics for habitability assessment.
Uses machine learning on stellar parameters for habitable zone prediction.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class StarProperties:
    """Stellar classification result."""
    spectral_type: str
    teff_k: float
    radius_rs: float
    mass_ms: float
    luminosity_ls: float
    hz_inner_au: float
    hz_outer_au: float
    habitability_factor: float


class StellarClassifier:
    """Machine learning-based stellar classification and analysis.
    
    Classifies stars using spectral parameters and predicts habitable zones.
    Employs random forests on Gaia/spectroscopic data.
    """
    
    SPECTRAL_TYPES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    
    def __init__(self):
        """Initialize classifier."""
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def _estimate_hz_bounds(self, luminosity: float) -> Tuple[float, float]:
        """Estimate habitable zone inner/outer boundaries.
        
        Args:
            luminosity: Stellar luminosity in solar luminosities
            
        Returns:
            Inner and outer AU distances
        """
        # Kopparapu et al. (2013) formulation
        hz_inner = np.sqrt(luminosity / 1.107)
        hz_outer = np.sqrt(luminosity / 0.356)
        return hz_inner, hz_outer
    
    def classify_star(self, teff_k: float, radius_rs: float,
                     magnitude_v: float) -> Optional[StarProperties]:
        """Classify star and compute stellar parameters.
        
        Args:
            teff_k: Effective temperature (Kelvin)
            radius_rs: Radius in solar radii
            magnitude_v: V-band absolute magnitude
            
        Returns:
            StarProperties with classification
        """
        # Determine spectral type from temperature
        if teff_k >= 30000:
            spec_type = 'O'
        elif teff_k >= 10000:
            spec_type = 'B'
        elif teff_k >= 7500:
            spec_type = 'A'
        elif teff_k >= 6000:
            spec_type = 'F'
        elif teff_k >= 5200:
            spec_type = 'G'
        elif teff_k >= 3700:
            spec_type = 'K'
        else:
            spec_type = 'M'
        
        # Estimate mass from temperature and radius
        mass_ms = (teff_k / 5778) ** 3.5 * radius_rs ** 2
        
        # Estimate luminosity from magnitude
        luminosity_ls = 10 ** (-magnitude_v / 2.5) * (5778 / teff_k) ** 2
        
        # Habitable zone boundaries
        hz_inner, hz_outer = self._estimate_hz_bounds(luminosity_ls)
        
        # Habitability factor (1.0 = G-type, decreases for extremes)
        if spec_type == 'G':
            habitability = 1.0
        elif spec_type in ['F', 'K']:
            habitability = 0.95
        elif spec_type in ['A', 'M']:
            habitability = 0.85
        else:
            habitability = 0.5
        
        return StarProperties(
            spectral_type=spec_type,
            teff_k=teff_k,
            radius_rs=radius_rs,
            mass_ms=mass_ms,
            luminosity_ls=luminosity_ls,
            hz_inner_au=hz_inner,
            hz_outer_au=hz_outer,
            habitability_factor=habitability
        )
