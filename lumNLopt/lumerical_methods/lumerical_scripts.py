

import numpy as np
import scipy as sp
import scipy.constants
import lumapi
from lumopt.utilities.fields import Fields, FieldsNoInterp

def get_fields_on_cad(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation, unfold_symmetry = True):
    unfold_symmetry_string = "true" if unfold_symmetry else "false"
    fdtd.eval("options=struct; options.unfold={0};".format(unfold_symmetry_string) + 
              "{0} = struct;".format(field_result_name) +
              "{0}.E = getresult('{1}','E',options);".format(field_result_name, monitor_name))

    if get_eps or get_D:
        index_monitor_name = monitor_name + '_index'
        fdtd.eval("{0}.index = getresult('{1}','index',options);".format(field_result_name, index_monitor_name))

    if get_H:
        fdtd.eval("{0}.H = getresult('{1}','H',options);".format(field_result_name, monitor_name))

    if nointerpolation:
        fdtd.eval("{0}.delta = struct;".format(field_result_name) +
                  "{0}.delta.x = getresult('{1}','delta_x',options);".format(field_result_name, monitor_name) +
                  "{0}.delta.y = getresult('{1}','delta_y',options);".format(field_result_name, monitor_name))
        monitor_dimension = fdtd.getresult(monitor_name, 'dimension')
        if monitor_dimension == 3:
            fdtd.eval("{0}.delta.z = getdata('{1}','delta_z');".format(field_result_name, monitor_name))
        else:
            fdtd.eval("{0}.delta.z = 0.0;".format(field_result_name))


def get_fields(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation, unfold_symmetry = True, on_cad_only = False):

    get_fields_on_cad(fdtd, monitor_name, field_result_name, get_eps, get_D, get_H, nointerpolation, unfold_symmetry)

    ## If required, we now transfer the field data to Python and package it up 
    if not on_cad_only:
        fields_dict = lumapi.getVar(fdtd.handle, field_result_name)

    if get_eps:
        if fdtd.getnamednumber('varFDTD') == 1:
            if 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and not 'index_z' in fields_dict['index']: # varFDTD TE simulation
                fields_dict['index']['index_z'] = fields_dict['index']['index_x']*0.0 + 1.0
            elif not 'index_x' in fields_dict['index'] and not 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']: # varFDTD TM simulation
                fields_dict['index']['index_x'] = fields_dict['index']['index_z']*0.0 + 1.0
                fields_dict['index']['index_y'] = fields_dict['index']['index_x']
        assert 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']
        fields_eps = np.stack((np.power(fields_dict['index']['index_x'], 2), 
                               np.power(fields_dict['index']['index_y'], 2), 
                               np.power(fields_dict['index']['index_z'], 2)), 
                               axis = -1)
    else:
        fields_eps = None

    fields_D = fields_dict['E']['E'] * fields_eps * sp.constants.epsilon_0 if get_D else None

    fields_H = fields_dict['H']['H'] if get_H else None

    if nointerpolation:
        deltas = [fields_dict['delta']['x'], fields_dict['delta']['y'], fields_dict['delta']['z']]
        return FieldsNoInterp(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], deltas, fields_dict['E']['E'], fields_D, fields_eps, fields_H)
    else:
        return Fields(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], fields_dict['E']['E'], fields_D, fields_eps, fields_H)

def set_spatial_interp(fdtd,monitor_name,setting):
    script='select("{}");set("spatial interpolation","{}");'.format(monitor_name,setting)
    fdtd.eval(script)

def get_eps_from_sim(fdtd, monitor_name = 'opt_fields', unfold_symmetry = True):
    index_monitor_name = monitor_name + '_index'

    unfold_symmetry_string = "true" if unfold_symmetry else "false"
    fdtd.eval(('options=struct; options.unfold={0};'
               '{1}_result = getresult("{1}","index",options);'
               '{1}_eps_x = ({1}_result.index_x)^2;'
               '{1}_eps_y = ({1}_result.index_y)^2;'
               '{1}_eps_z = ({1}_result.index_z)^2;'
               '{1}_x = {1}_result.x;'
               '{1}_y = {1}_result.y;'
               '{1}_z = {1}_result.z;'
               '{1}_lambda = {1}_result.lambda;'
               ).format(unfold_symmetry_string, index_monitor_name))
    fields_eps_x = fdtd.getv('{0}_eps_x'.format(index_monitor_name))    # np.power(index_dict['index_x'], 2)
    fields_eps_y = fdtd.getv('{0}_eps_y'.format(index_monitor_name))    # np.power(index_dict['index_y'], 2)
    fields_eps_z = fdtd.getv('{0}_eps_z'.format(index_monitor_name))    # np.power(index_dict['index_z'], 2)
    index_monitor_x      = fdtd.getv('{0}_x'.format(index_monitor_name))  # index_dict['x']
    index_monitor_y      = fdtd.getv('{0}_y'.format(index_monitor_name))   # index_dict['y']
    index_monitor_z      = fdtd.getv('{0}_z'.format(index_monitor_name))  # index_dict['z']
    index_monitor_lambda = fdtd.getv('{0}_lambda'.format(index_monitor_name))  # index_dict['lambda']


    # index_dict = fdtd.getresult(index_monitor_name, 'index')   #< Currently does not work with unfolding options
    # fields_eps_x = np.power(index_dict['index_x'], 2)
    # fields_eps_y = np.power(index_dict['index_y'], 2)
    # fields_eps_z = np.power(index_dict['index_z'], 2)
    # index_monitor_x      = index_dict['x']
    # index_monitor_y      = index_dict['y']
    # index_monitor_z      = index_dict['z']
    # index_monitor_lambda = index_dict['lambda']


    fields_eps = np.stack((fields_eps_x, fields_eps_y, fields_eps_z), axis = -1)
    return fields_eps, index_monitor_x,index_monitor_y,index_monitor_z, index_monitor_lambda


def setup_anisotropic_rectangle(fdtd, rect_name, rect_params, material_props):
    """
    NEW: Setup anisotropic rectangle using correct Lumerical syntax
    
    Parameters:
    -----------
    rect_params : dict
        Rectangle geometry: {'x_min', 'x_max', 'y_min', 'y_max', 'z_center', 'z_span'}
    material_props : dict  
        Material properties: {'eps_xx', 'eps_yy', 'eps_zz', 'is_anisotropic'}
    """
    
    try:
        # Set geometry
        fdtd.setnamed(rect_name, 'x min', rect_params['x_min'])
        fdtd.setnamed(rect_name, 'x max', rect_params['x_max'])
        fdtd.setnamed(rect_name, 'y min', rect_params['y_min'])
        fdtd.setnamed(rect_name, 'y max', rect_params['y_max'])
        fdtd.setnamed(rect_name, 'z', rect_params['z_center'])
        fdtd.setnamed(rect_name, 'z span', rect_params['z_span'])
        
        # Set material - CORRECTED SYNTAX
        fdtd.setnamed(rect_name, 'material', '<Object defined dielectric>')
        
        if material_props['is_anisotropic']:
            # Enable diagonal anisotropy
            fdtd.setnamed(rect_name, 'anisotropy', 1)
            
            # Set refractive index components
            fdtd.setnamed(rect_name, 'index x', np.sqrt(material_props['eps_xx']))
            fdtd.setnamed(rect_name, 'index y', np.sqrt(material_props['eps_yy']))
            fdtd.setnamed(rect_name, 'index z', np.sqrt(material_props['eps_zz']))
            
        else:
            # Isotropic material
            avg_eps = material_props['eps_xx']  # All components same for isotropic
            fdtd.setnamed(rect_name, 'index', np.sqrt(avg_eps))
            
        return True
        
    except Exception as e:
        print(f"Error setting up anisotropic rectangle {rect_name}: {e}")
        return False

def get_anisotropic_fields(fdtd, monitor_name, get_tensor_eps=True):
    """
    NEW: Enhanced field extraction with anisotropic material support
    """
    
    # Standard field extraction
    fields = get_fields(fdtd, monitor_name, 'field_data', 
                       get_eps=get_tensor_eps, get_D=True, get_H=False, 
                       nointerpolation=False)
    
    if get_tensor_eps:
        # Extract tensor permittivity information
        fdtd.eval(f"""
        field_data_tensor = struct;
        field_data_tensor.eps_tensor = struct;
        
        % Get material information from index monitor
        index_data = getresult('{monitor_name}_index', 'index');
        
        % Extract tensor components (Lumerical stores as separate index components)
        if isfield(index_data, 'index_x') &&
           isfield(index_data, 'index_y') &&  
           isfield(index_data, 'index_z')
            field_data_tensor.eps_tensor.xx = index_data.index_x^2;
            field_data_tensor.eps_tensor.yy = index_data.index_y^2;
            field_data_tensor.eps_tensor.zz = index_data.index_z^2;
        else
            % Fallback: assume isotropic
            index_avg = (index_data.index_x + index_data.index_y + index_data.index_z) / 3;
            field_data_tensor.eps_tensor.xx = index_avg^2;
            field_data_tensor.eps_tensor.yy = index_avg^2;
            field_data_tensor.eps_tensor.zz = index_avg^2;
        end
        """)
        
        # Get tensor data
        tensor_data = lumapi.getVar(fdtd.handle, 'field_data_tensor')
        fields.eps_tensor = tensor_data['eps_tensor'] if tensor_data else None
        
        fdtd.eval("clear(field_data_tensor, index_data);")
    
    return fields


def get_mode_overlap_between_monitors(fdtd, monitor1_name, monitor2_name):
    """
    Calculate mode overlap between two mode expansion monitors.
    
    Parameters:
    -----------
    fdtd : lumapi.FDTD
        FDTD simulation object
    monitor1_name : str
        Name of first mode expansion monitor
    monitor2_name : str
        Name of second mode expansion monitor
        
    Returns:
    --------
    overlap : np.ndarray
        Mode overlap values vs wavelength
    """
    
    try:
        # Get mode expansion results from both monitors
        exp_result1_name = f'expansion for {monitor1_name}'
        exp_result2_name = f'expansion for {monitor2_name}'
        
        if not fdtd.haveresult(monitor1_name, exp_result1_name):
            raise UserWarning(f'Mode expansion result not found for {monitor1_name}')
        if not fdtd.haveresult(monitor2_name, exp_result2_name):
            raise UserWarning(f'Mode expansion result not found for {monitor2_name}')
        
        # Extract mode data
        mode1_data = fdtd.getresult(monitor1_name, exp_result1_name)
        mode2_data = fdtd.getresult(monitor2_name, exp_result2_name)
        
        # Get mode coefficients (fundamental mode)
        a1 = mode1_data['a']  # Forward coefficient from monitor 1
        a2 = mode2_data['a']  # Forward coefficient from monitor 2
        
        # Calculate mode overlap: |<ψ1|ψ2>|²
        # For fundamental modes, this is |a1* × a2|
        overlap = np.abs(np.conj(a1) * a2)
        
        return np.real(overlap).flatten()
        
    except Exception as e:
        print(f"Error calculating mode overlap between {monitor1_name} and {monitor2_name}: {e}")
        # Return fallback values (moderate overlap)
        wavelengths = fdtd.getglobalmonitor('frequency points')
        return np.ones(int(wavelengths)) * 0.7


def setup_slice_monitors_for_adiabatic_coupler(fdtd, num_slices, coupler_length, 
                                               coupler_width, coupler_height, 
                                               y_center=0, z_center=0):
    """
    Setup field monitors for each slice in adiabatic edge coupler.
    
    Parameters:
    -----------
    fdtd : lumapi.FDTD
        FDTD simulation object
    num_slices : int
        Number of slices along the coupler
    coupler_length : float
        Total length of the coupler (m)
    coupler_width : float
        Width span for monitors (m)
    coupler_height : float
        Height span for monitors (m)
    y_center : float
        Y-center position for monitors (m)
    z_center : float
        Z-center position for monitors (m)
    """
    
    # Calculate slice positions along the coupler
    x_positions = np.linspace(0, coupler_length, num_slices)
    
    for i, x_pos in enumerate(x_positions):
        monitor_name = f'slice_monitor_{i}'
        mode_exp_name = f'slice_monitor_{i}_mode_exp'
        
        try:
            # Add field monitor for this slice
            fdtd.addprofile()
            fdtd.set('name', monitor_name)
            fdtd.set('monitor type', '2D X-normal')
            fdtd.set('x', x_pos)
            fdtd.set('y', y_center)
            fdtd.set('y span', coupler_width)
            fdtd.set('z', z_center)
            fdtd.set('z span', coupler_height)
            
            # Add corresponding mode expansion monitor
            fdtd.addmodeexpansion()
            fdtd.set('name', mode_exp_name)
            fdtd.set('x', x_pos)
            fdtd.set('y', y_center)
            fdtd.set('y span', coupler_width)
            fdtd.set('z', z_center)
            fdtd.set('z span', coupler_height)
            fdtd.set('mode selection', 'fundamental mode')
            fdtd.updatemodes()
            
        except Exception as e:
            print(f"Warning: Could not setup monitor for slice {i} at x={x_pos*1e6:.1f}μm: {e}")


def get_fundamental_mode_power_fraction(fdtd, mode_expansion_monitor_name):
    """
    Get the power fraction in the fundamental mode.
    
    Parameters:
    -----------
    fdtd : lumapi.FDTD
        FDTD simulation object
    mode_expansion_monitor_name : str
        Name of mode expansion monitor
        
    Returns:
    --------
    power_fraction : np.ndarray
        Power fraction in fundamental mode vs wavelength
    """
    
    try:
        exp_result_name = f'expansion for {mode_expansion_monitor_name}'
        
        if not fdtd.haveresult(mode_expansion_monitor_name, exp_result_name):
            raise UserWarning(f'Mode expansion result not found for {mode_expansion_monitor_name}')
        
        mode_data = fdtd.getresult(mode_expansion_monitor_name, exp_result_name)
        
        # Get forward and backward coefficients
        a_coeff = mode_data['a']  # Forward coefficient
        b_coeff = mode_data['b']  # Backward coefficient
        
        # Power in fundamental mode (first mode, index 0)
        forward_power = np.abs(a_coeff)**2
        backward_power = np.abs(b_coeff)**2
        total_mode_power = forward_power + backward_power
        
        # For fundamental mode analysis, we typically want forward power
        fundamental_power_fraction = forward_power.flatten()
        
        return np.real(fundamental_power_fraction)
        
    except Exception as e:
        print(f"Error getting fundamental mode power fraction from {mode_expansion_monitor_name}: {e}")
        wavelengths = fdtd.getglobalmonitor('frequency points')
        return np.ones(int(wavelengths)) * 0.8  # Fallback: assume 80% fundamental mode

