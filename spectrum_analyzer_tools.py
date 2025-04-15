import numpy as np
import xml.etree.ElementTree as ET



def _find_parameter(element, Id):
    parameter_ids = [parameter.attrib['Id'] for parameter in element.findall('Parameter')]
    return element.findall('Parameter')[ parameter_ids.index(Id) ].attrib['Value']


def get_trace(file, spec_density=False):
    """
    Imports a spectrum trace saved by the U3751 spectrum analyzer, which outputs traces saved as .xml files.
    The argument to the function is the path to the xml file.
    
    If spec_density is False, returns the saved noise amplitude.
    If spec_density is True, returns inferred noise spectral density calculated using the resolution bandwidth,
        which is also saved in the .xml file.
    """
        
    tree = ET.parse(file)
    root = tree.getroot()
        
    data = root.find('Context').find('Trace').find('Table')
    amp = np.zeros(len(data))
    for elem in data:
        amp[int(elem.attrib['Point'])] = (elem.attrib['Lvl'])
    
    
    f_min = float(_find_parameter(root.find('Context'), 'FA')) / 1e6
    f_max = float(_find_parameter(root.find('Context'), 'FB')) / 1e6
    freq = np.linspace(f_min, f_max, len(amp))

    if spec_density:
        rbw = float(_find_parameter(root.find('Context'), 'RB'))
        specdens = amp - 10*np.log10(rbw)
        return (freq, specdens)
    else:
        return (freq, amp)
    
    
def get_rbw(file):
    """Returns the resolution bandwidth of a trace saved by the spectrum analyzer (in Hz)"""
    tree = ET.parse(file)
    root = tree.getroot()
    return float(_find_parameter(root.find('Context'), 'RB'))