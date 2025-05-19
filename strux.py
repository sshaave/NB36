import xml.etree.ElementTree as ET
import uuid
from xml.dom import minidom

def compute_average_z(contour):
    """
    Compute the arithmetic average of all z coordinates from <point> elements within the given contour element.
    """
    z_values = []
    for point in contour.findall('.//point'):
        try:
            z_val = float(point.get('z', '0'))
            z_values.append(z_val)
        except ValueError:
            pass
    if z_values:
        return sum(z_values) / len(z_values)
    else:
        return 0.0

def process_input(xml_fragment):
    """
    Process the given XML fragment (which contains many <slab> elements) and create corresponding
    <surface_reinforcement_parameters> elements. The input is wrapped in a root element.
    """
    # Wrap the entire fragment with a root element to form valid XML
    wrapped_xml = f"<root>{xml_fragment}</root>"
    root = ET.fromstring(wrapped_xml)
    
    srp_elements = []
    
    for slab in root.findall('slab'):
        # Get slab-level attributes
        slab_last_change = slab.get('last_change', '')
        slab_action = slab.get('action', 'added')
        
        # Find the first <slab_part> element
        slab_part = slab.find('slab_part')
        if slab_part is None:
            continue
        
        # Use the slab_part's guid for the base_shell element
        base_shell_guid = slab_part.get('guid', '')
        
        # Locate the <contour> element in the slab_part (it might be nested)
        contour = slab_part.find('.//contour')
        if contour is None:
            continue
        
        # Calculate the average z value from all <point> elements in the contour
        avg_z = compute_average_z(contour)
        
        # Create the center attributes:
        # x and y are forced; z is calculated (formatted to preserve precision)
        center_attrib = {
            "x": "25.037",
            "y": "26.045",
            "z": f"{avg_z:.9f}",
            "polar_system": "true"
        }
        
        # Create the new <surface_reinforcement_parameters> element.
        srp = ET.Element("surface_reinforcement_parameters", {
            "guid": str(uuid.uuid4()),
            "last_change": slab_last_change,
            "action": slab_action
        })
        
        # Append the child elements.
        ET.SubElement(srp, "base_shell", {"guid": base_shell_guid})
        ET.SubElement(srp, "center", center_attrib)
        ET.SubElement(srp, "x_direction", {"x": "1", "y": "0", "z": "0"})
        ET.SubElement(srp, "y_direction", {"x": "0", "y": "1", "z": "0"})
        
        srp_elements.append(srp)
    
    return srp_elements

def pretty_print(elem):
    """
    Return a pretty-printed XML string for the given Element using tab indentation.
    Ensures that empty tags are expanded (e.g. <tag></tag>).
    """
    rough_string = ET.tostring(elem, 'utf-8', short_empty_elements=False)
    reparsed = minidom.parseString(rough_string)
    pretty = reparsed.toprettyxml(indent="\t")
    # Remove XML declaration and any empty lines.
    return "\n".join(line for line in pretty.splitlines() if line.strip() and not line.startswith("<?xml"))

def generate_output(srp_elements):
    """
    Combine the pretty-printed XML strings for each generated <surface_reinforcement_parameters> element.
    """
    output = ""
    for srp in srp_elements:
        output += pretty_print(srp) + "\n"
    return output

def main():
    # Read input from a text file containing many <slab> elements.
    with open("input.txt", "r") as infile:
        input_data = infile.read()
    
    # Process the input data to generate the new reinforcement parameters elements.
    srp_elements = process_input(input_data)
    output_xml = generate_output(srp_elements)
    
    # Write the output to a file.
    with open("output.txt", "w") as outfile:
        outfile.write(output_xml)
    
    print("Processing complete. Output written to output.txt.")

if __name__ == "__main__":
    main()
