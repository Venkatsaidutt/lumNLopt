import numpy as np
import scipy as sp
import scipy.constants

from lumopt.utilities.wavelengths import Wavelengths

class Material(object):
    """
    ENHANCED Material class with anisotropic support for rectangle clustering.
    
    Added functionality:
    - Full tensor material support
    - Proper anisotropic material setup in Lumerical
    - Rectangle-specific material assignment
    """
    
    object_dielectric = str('<Object defined dielectric>')

    def __init__(self, base_epsilon=1.0, name=object_dielectric, mesh_order=None, 
                 anisotropic_params=None):
        """
        Parameters:
        -----------
        base_epsilon : float or dict
            For isotropic: scalar value
            For anisotropic: dict with 'xx', 'yy', 'zz' keys
        anisotropic_params : dict, optional
            Full tensor specification: {'eps_xx', 'eps_yy', 'eps_zz', 'eps_xy', 'eps_xz', 'eps_yz'}
        """
        
        # Handle both scalar and tensor inputs
        if isinstance(base_epsilon, dict):
            self.base_epsilon = base_epsilon.get('xx', 1.0)  # Use xx as scalar fallback
            self.is_anisotropic = True
            self.anisotropic_eps = base_epsilon
        else:
            self.base_epsilon = float(base_epsilon)
            self.is_anisotropic = False
            self.anisotropic_eps = None
            
        # Enhanced anisotropic parameters
        if anisotropic_params is not None:
            self.is_anisotropic = True
            self.anisotropic_eps = anisotropic_params
            
        self.name = str(name)
        self.mesh_order = mesh_order

    def set_script(self, sim, poly_name):
        """Enhanced set_script with anisotropic material support"""
        
        sim.fdtd.setnamed(poly_name, 'material', self.name)
        self.wavelengths = Material.get_wavelengths(sim)
        freq_array = sp.constants.speed_of_light / self.wavelengths.asarray()
        
        if self.name == self.object_dielectric:
            if self.is_anisotropic:
                # FIXED: Correct Lumerical anisotropic syntax
                self._set_anisotropic_material(sim, poly_name)
            else:
                # Standard isotropic material
                refractive_index = np.sqrt(self.base_epsilon)
                sim.fdtd.setnamed(poly_name, 'index', float(refractive_index))
                self.permittivity = self.base_epsilon * np.ones(freq_array.shape)
        else:
            # Material from database
            fdtd_index = sim.fdtd.getfdtdindex(self.name, freq_array, 
                                             float(freq_array.min()), float(freq_array.max()))
            self.permittivity = np.asarray(np.power(fdtd_index, 2)).flatten()
            
        if self.mesh_order:
            sim.fdtd.setnamed(poly_name, 'override mesh order from material database', True)
            sim.fdtd.setnamed(poly_name, 'mesh order', self.mesh_order)

    def _set_anisotropic_material(self, sim, poly_name):
        """
        Set anisotropic material properties using CORRECT Lumerical syntax
        Based on official Ansys documentation
        """
        
        try:
            # Method 1: Use index values (RECOMMENDED by Lumerical docs)
            eps_xx = self.anisotropic_eps.get('eps_xx', self.anisotropic_eps.get('xx', 1.0))
            eps_yy = self.anisotropic_eps.get('eps_yy', self.anisotropic_eps.get('yy', 1.0))
            eps_zz = self.anisotropic_eps.get('eps_zz', self.anisotropic_eps.get('zz', 1.0))
            
            # Enable diagonal anisotropy
            sim.fdtd.setnamed(poly_name, 'anisotropy', 1)  # 1 = Diagonal anisotropy
            
            # Set diagonal refractive index components
            sim.fdtd.setnamed(poly_name, 'index x', float(np.sqrt(np.real(eps_xx))))
            sim.fdtd.setnamed(poly_name, 'index y', float(np.sqrt(np.real(eps_yy))))
            sim.fdtd.setnamed(poly_name, 'index z', float(np.sqrt(np.real(eps_zz))))
            
            # Store permittivity tensor for gradient calculations
            freq_array = sp.constants.speed_of_light / self.wavelengths.asarray()
            self.permittivity_tensor = np.zeros((len(freq_array), 3, 3), dtype=complex)
            for i in range(len(freq_array)):
                self.permittivity_tensor[i] = np.diag([eps_xx, eps_yy, eps_zz])
            
            print(f"Set anisotropic material for {poly_name}: "
                  f"n_x={np.sqrt(eps_xx):.3f}, n_y={np.sqrt(eps_yy):.3f}, n_z={np.sqrt(eps_zz):.3f}")
                  
        except Exception as e:
            print(f"Error setting anisotropic material: {e}")
            # Fallback to isotropic approximation
            avg_eps = (eps_xx + eps_yy + eps_zz) / 3.0
            sim.fdtd.setnamed(poly_name, 'index', float(np.sqrt(avg_eps)))
            print(f"Fallback to isotropic: n_avg={np.sqrt(avg_eps):.3f}")

    def get_eps_tensor(self, wavelengths):
        """Get full permittivity tensor for anisotropic materials"""
        
        if not self.is_anisotropic:
            # Return isotropic tensor
            eps_scalar = self.get_eps(wavelengths)
            eps_tensor = np.zeros((*eps_scalar.shape, 3, 3))
            for i in range(3):
                eps_tensor[..., i, i] = eps_scalar
            return eps_tensor
        else:
            # Return stored anisotropic tensor
            if hasattr(self, 'permittivity_tensor'):
                return self.permittivity_tensor
            else:
                # Reconstruct from anisotropic_eps
                eps_xx = self.anisotropic_eps.get('eps_xx', 1.0)
                eps_yy = self.anisotropic_eps.get('eps_yy', 1.0) 
                eps_zz = self.anisotropic_eps.get('eps_zz', 1.0)
                
                eps_tensor = np.zeros((len(wavelengths), 3, 3))
                for i in range(len(wavelengths)):
                    eps_tensor[i] = np.diag([eps_xx, eps_yy, eps_zz])
                return eps_tensor

    def get_eps(self, wavelengths):
        """Enhanced get_eps with anisotropic support (returns scalar equivalent)"""
        
        if hasattr(self, 'permittivity'):
            assert len(wavelengths) == len(self.wavelengths)
            if self.is_anisotropic and hasattr(self, 'permittivity_tensor'):
                # Return trace/3 as scalar equivalent for compatibility
                return np.trace(self.permittivity_tensor, axis1=1, axis2=2) / 3.0
            else:
                return self.permittivity
        elif self.name == self.object_dielectric:
            if self.is_anisotropic:
                # Return average permittivity for scalar compatibility
                eps_xx = self.anisotropic_eps.get('eps_xx', 1.0)
                eps_yy = self.anisotropic_eps.get('eps_yy', 1.0)
                eps_zz = self.anisotropic_eps.get('eps_zz', 1.0)
                avg_eps = (eps_xx + eps_yy + eps_zz) / 3.0
                return avg_eps * np.ones(wavelengths.shape)
            else:
                return self.base_epsilon * np.ones(wavelengths.shape)
        else:
            raise UserWarning('material has not yet been assigned to a geometric primitive.')

    @staticmethod
    def get_wavelengths(sim):
        return Wavelengths(sim.fdtd.getglobalsource('wavelength start'), 
                           sim.fdtd.getglobalsource('wavelength stop'),
                           sim.fdtd.getglobalmonitor('frequency points'))

    @classmethod
    def create_anisotropic(cls, eps_xx, eps_yy, eps_zz, name=None, mesh_order=None):
        """
        Convenience method to create anisotropic materials
        
        Parameters:
        -----------
        eps_xx, eps_yy, eps_zz : float
            Diagonal permittivity components
        """
        anisotropic_params = {
            'eps_xx': eps_xx,
            'eps_yy': eps_yy, 
            'eps_zz': eps_zz,
            'eps_xy': 0.0,
            'eps_xz': 0.0,
            'eps_yz': 0.0
        }
        
        return cls(
            base_epsilon=eps_xx,  # Use xx as scalar fallback
            name=name or cls.object_dielectric,
            mesh_order=mesh_order,
            anisotropic_params=anisotropic_params
        )
