import pandas as pd
import numpy as np
import os
import gc

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_peaks_path = os.path.join(base_dir, 'hmdb_raw_peaks.csv')
    old_library_path = r'd:\hack\BDI\NMRQNet\data\simulation_data\reference_library_38.csv'
    output_path = os.path.join(base_dir, 'hmdb_reference_library.csv')
    
    print("Loading old library...")
    df_old = pd.read_csv(old_library_path)
    if 'Unnamed: 0' in df_old.columns:
        df_old = df_old.drop(columns=['Unnamed: 0'])
    
    old_metabolites = set(df_old['Name'].unique())
    print(f"Loaded {len(old_metabolites)} metabolites from old library.")
    
    if not os.path.exists(raw_peaks_path):
        print(f"Error: Could not find {raw_peaks_path}.")
        return
        
    print("Processing new HMDB raw peaks in chunks to save memory...")
    
    # We use all metabolites since they are now pre-filtered by 'quantified' status in parsing
    selected_metabolites = set()
    all_chunks = []
    
    for chunk in pd.read_csv(raw_peaks_path, chunksize=500_000, engine='c'):
        chunk_filtered = chunk[~chunk['metabolite_name'].isin(old_metabolites)]
        
        # Keep all unique metabolites from this chunk
        unique_in_chunk = chunk_filtered['metabolite_name'].unique()
        for m in unique_in_chunk:
            selected_metabolites.add(m)
                
        chunk_kept = chunk_filtered
        
        if not chunk_kept.empty:
            df_fmt = pd.DataFrame({
                'Name': chunk_kept['metabolite_name'],
                'cluster': chunk_kept['peak_ppm'].round(1),
                'loc': chunk_kept['peak_ppm'],
                'height': chunk_kept['peak_height'],
                'width': chunk_kept['peak_width']
            })
            all_chunks.append(df_fmt)
            
            
    print(f"Selected {len(selected_metabolites)} new 'quantified' metabolites from HMDB.")
    
    print("Concatenating and sorting...")
    if all_chunks:
        df_new_formatted = pd.concat(all_chunks, ignore_index=True)
        df_combined = pd.concat([df_old, df_new_formatted], ignore_index=True)
    else:
        df_combined = df_old
        
    df_combined = df_combined.sort_values(by=['Name', 'loc']).reset_index(drop=True)
    
    print(f"Saving to {output_path}...")
    df_combined.to_csv(output_path, index=True, index_label="")
    
    total_metabolites = df_combined['Name'].nunique()
    print(f"Successfully generated {output_path}")
    print(f"Total metabolites in final library: {total_metabolites}")

if __name__ == '__main__':
    main()
