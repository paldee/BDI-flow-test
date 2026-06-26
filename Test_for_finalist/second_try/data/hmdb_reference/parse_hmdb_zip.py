import os
import zipfile
import lxml.etree as ET
import pandas as pd
from tqdm import tqdm

def get_metabolite_mapping(metabolites_zip_path):
    """
    Parse hmdb_metabolites.zip to get a mapping of HMDB ID to Name, 
    and optionally filter by biofluid.
    Since hmdb_metabolites.xml is huge (~3.5GB unzipped), we use iterparse.
    """
    print(f"Parsing {metabolites_zip_path} for metabolite names...")
    mapping = {}
    
    # We use iterparse to avoid loading the whole 3.5GB XML into RAM
    with zipfile.ZipFile(metabolites_zip_path, 'r') as z:
        # Typically there is one big xml file inside
        xml_filename = z.namelist()[0]
        
        with z.open(xml_filename) as f:
            context = ET.iterparse(f, events=('end',), tag='{http://www.hmdb.ca}metabolite')
            for event, elem in context:
                accession_elem = elem.find('{http://www.hmdb.ca}accession')
                name_elem = elem.find('{http://www.hmdb.ca}name')
                status_elem = elem.find('{http://www.hmdb.ca}status')
                
                is_quantified = (status_elem is not None and status_elem.text == 'quantified')
                
                # Check biological properties to ensure it's found in humans (e.g. blood, urine, etc)
                # This helps reduce the database to relevant metabolites
                is_human = False
                bio_props = elem.find('{http://www.hmdb.ca}biological_properties')
                if bio_props is not None:
                    biospecimens = bio_props.find('{http://www.hmdb.ca}biospecimen_locations')
                    if biospecimens is not None:
                        for spec in biospecimens.findall('{http://www.hmdb.ca}biospecimen'):
                            if spec.text and spec.text.lower() in ['blood', 'urine', 'saliva', 'cerebrospinal fluid (csf)', 'feces', 'sweat', 'breast milk', 'cellular cytoplasm']:
                                is_human = True
                                break
                
                if accession_elem is not None and name_elem is not None and is_human and is_quantified:
                    mapping[accession_elem.text] = name_elem.text
                    
                # Clean up to save memory
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                    
    print(f"Found {len(mapping)} human-related metabolites in the database.")
    return mapping

def extract_spectra_peaks(spectra_zip_path, mapping):
    """
    Parse hmdb_nmr_spectra.zip to extract 1H NMR peaks for valid HMDB IDs.
    """
    print(f"Parsing {spectra_zip_path} for 1H NMR spectra...")
    all_peaks = []
    
    with zipfile.ZipFile(spectra_zip_path, 'r') as z:
        names = z.namelist()
        for name in tqdm(names, desc="Extracting Spectra"):
            if not name.endswith('.xml'):
                continue
                
            with z.open(name) as f:
                try:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    
                    nucleus = root.find('nucleus')
                    if nucleus is None or '1H' not in nucleus.text:
                        continue
                        
                    db_id_elem = root.find('database-id')
                    if db_id_elem is None or db_id_elem.text not in mapping:
                        continue
                        
                    hmdb_id = db_id_elem.text
                    metabolite_name = mapping[hmdb_id]
                    
                    peaks_elem = root.find('nmr-one-d-peaks')
                    if peaks_elem is not None:
                        for peak in peaks_elem.findall('nmr-one-d-peak'):
                            shift_elem = peak.find('chemical-shift')
                            intensity_elem = peak.find('intensity')
                            
                            if shift_elem is not None and shift_elem.text:
                                try:
                                    ppm = float(shift_elem.text)
                                    intensity = float(intensity_elem.text) if intensity_elem is not None and intensity_elem.text else 1.0
                                    all_peaks.append({
                                        'metabolite_name': metabolite_name,
                                        'peak_ppm': ppm,
                                        'peak_height': intensity,
                                        'peak_width': 1.0 # Default
                                    })
                                except ValueError:
                                    pass
                except ET.ParseError:
                    continue
                    
    return all_peaks

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metabolites_zip = os.path.join(base_dir, 'hmdb_metabolites.zip')
    spectra_zip = os.path.join(base_dir, 'hmdb_nmr_spectra.zip')
    output_raw_csv = os.path.join(base_dir, 'hmdb_raw_peaks.csv')
    
    if not os.path.exists(metabolites_zip) or not os.path.exists(spectra_zip):
        print("Error: Missing required ZIP files.")
        return
        
    mapping = get_metabolite_mapping(metabolites_zip)
    peaks = extract_spectra_peaks(spectra_zip, mapping)
    
    if peaks:
        df = pd.DataFrame(peaks)
        df.to_csv(output_raw_csv, index=False)
        print(f"Successfully extracted {len(df)} peaks from {df['metabolite_name'].nunique()} metabolites.")
        print(f"Saved to {output_raw_csv}")
        print("You can now run build_reference_spectra.py to merge this with the 38 baseline metabolites.")
    else:
        print("No valid 1H NMR spectra found for the selected metabolites.")

if __name__ == '__main__':
    main()
