import numpy as np
import pandas as pd
import os
import argparse
import shutil
from tqdm import tqdm

def lorentzian(x, x0, a, w):
    return a / (1 + ((x - x0) / (w / 2))**2)

def generate_realistic_dataset(num_samples=10000, output_dir='.'):
    ppm_values = np.round(np.arange(0.0, 10.001, 0.001), 3)
    
    # 1. Load the reference library
    ref_lib_path = os.path.join(output_dir, 'hmdb_reference', 'hmdb_reference_library.csv')
    if not os.path.exists(ref_lib_path):
        print(f"Error: Could not find {ref_lib_path}")
        return
        
    df_lib = pd.read_csv(ref_lib_path)
    
    metabolites = df_lib['Name'].unique()
    print(f"Loaded {len(metabolites)} metabolites from the reference library. Pre-calculating spectra...")
    pure_spectra = {}
    
    # Vectorized calculation per metabolite
    for meta in tqdm(metabolites, desc="Calculating pure spectra"):
        meta_peaks = df_lib[df_lib['Name'] == meta]
        
        locs = meta_peaks['loc'].values[:, np.newaxis]
        heights = meta_peaks['height'].values[:, np.newaxis]
        widths = (meta_peaks['width'].values / 450.0)[:, np.newaxis]
        
        # ppm_values: (10001,) -> (1, 10001)
        x = ppm_values[np.newaxis, :]
        
        # specs: (N_peaks, 10001)
        specs = heights / (1 + ((x - locs) / (widths / 2))**2)
        spec = np.sum(specs, axis=0)
            
        area = np.trapezoid(spec, ppm_values)
        if area > 0:
            spec = spec / area
        pure_spectra[meta] = spec

    data = {'ppm': ppm_values}
    ground_truth = []
    
    np.random.seed(42)
    print(f"Generating dataset with {num_samples} samples... This might take a moment.")
    
    total_metabolites = len(metabolites)
    
    for i in tqdm(range(1, num_samples + 1)):
        sample_intensity = np.zeros_like(ppm_values)
        
        # Poisson distribution for number of metabolites, clipped between 3 and 50 (or total available)
        max_metabolites = min(50, total_metabolites)
        num_metabolites = int(np.clip(np.random.poisson(15), 3, max_metabolites))
        
        chosen_metabolites = np.random.choice(metabolites, num_metabolites, replace=False)
        
        gt_row = {'Sample_ID': f'Sample_{i:05d}'}
        for meta in metabolites:
            gt_row[meta] = 0.0
            
        for meta in chosen_metabolites:
            concentration = np.random.gamma(shape=2.0, scale=1.0)
            
            # Residual chemical shift drift (±0.01-0.05 ppm is a bit large, let's stick to small drift as per instructions)
            # The instructions said: "คง residual shift ไว้ (σ = 0.0005-0.001 ppm) เพื่อจำลอง drift ที่เหลือจาก pH"
            shift_error = np.random.normal(0, 0.0005)
            shift_idx = int(round(shift_error / 0.001))
            
            spec = pure_spectra[meta] * concentration
            
            if shift_idx > 0:
                spec = np.concatenate((np.zeros(shift_idx), spec[:-shift_idx]))
            elif shift_idx < 0:
                spec = np.concatenate((spec[-shift_idx:], np.zeros(-shift_idx)))
                
            sample_intensity += spec
            gt_row[meta] = round(concentration, 4)
            
        # Add very tiny noise (no big baseline drift as data is pre-cleaned)
        noise = np.random.normal(0, 0.01, len(ppm_values))
        sample_intensity += noise
        sample_intensity = np.clip(sample_intensity, 0, None)
        
        data[f'Sample_{i:05d}'] = np.round(sample_intensity, 4)
        ground_truth.append(gt_row)

    print("Converting to DataFrames...")
    df_data = pd.DataFrame(data)
    df_gt = pd.DataFrame(ground_truth)
    
    output_csv = os.path.join(output_dir, 'mock_nmr_data_10k.csv')
    gt_csv = os.path.join(output_dir, 'mock_ground_truth_10k.csv')
    
    print(f"Saving data to {output_csv}...")
    df_data.to_csv(output_csv, index=False)
    
    print(f"Saving ground truth to {gt_csv}...")
    df_gt.to_csv(gt_csv, index=False)
    
    print(f"Successfully generated {output_csv}")
    print(f"Successfully generated {gt_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    args = parser.parse_args()
    
    generate_realistic_dataset(num_samples=args.samples, output_dir=args.output_dir)
