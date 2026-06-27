import numpy as np
import pandas as pd
import os
import argparse
import shutil

def lorentzian(x, x0, a, w):
    """
    x: ppm axis
    x0: location
    a: height
    w: width
    """
    return a / (1 + ((x - x0) / (w / 2))**2)

def generate_realistic_dataset(num_samples=1000, output_dir='.'):
    # Define the ppm axis: 0 to 10 ppm with 0.001 step (10,001 points)
    ppm_values = np.round(np.arange(0.0, 10.001, 0.001), 3)
    
    # 1. Load the reference library
    ref_lib_path = r'd:\hack\BDI\NMRQNet\data\simulation_data\reference_library_38.csv'
    if not os.path.exists(ref_lib_path):
        print(f"Error: Could not find {ref_lib_path}")
        return
        
    df_lib = pd.read_csv(ref_lib_path)
    
    # Copy the reference library to the output directory
    ref_dir = os.path.join(output_dir, 'reference_library')
    os.makedirs(ref_dir, exist_ok=True)
    shutil.copy(ref_lib_path, os.path.join(ref_dir, 'reference_library_38.csv'))
    
    # Group peaks by metabolite
    metabolites = df_lib['Name'].unique()
    print(f"Loaded {len(metabolites)} metabolites from the reference library.")
    
    # Pre-calculate pure spectrum for each metabolite to save time
    pure_spectra = {}
    for meta in metabolites:
        meta_peaks = df_lib[df_lib['Name'] == meta]
        spec = np.zeros_like(ppm_values)
        for _, row in meta_peaks.iterrows():
            loc = row['loc']
            height = row['height']
            width = row['width']
            # Using scaler = 450 as a default high-resolution scaler (from NMRQNet)
            scaler = 450.0
            W = width / scaler
            spec += lorentzian(ppm_values, loc, height, W)
            
        # Normalize the area under the curve to 1
        area = np.trapezoid(spec, ppm_values)
        if area > 0:
            spec = spec / area
        pure_spectra[meta] = spec

    data = {'ppm': ppm_values}
    ground_truth = []
    
    np.random.seed(42)
    print(f"Generating dataset with {num_samples} samples... This might take a moment.")
    
    for i in range(1, num_samples + 1):
        sample_intensity = np.zeros_like(ppm_values)
        
        # Decide how many metabolites to include in this sample (e.g., 10 to 38)
        num_metabolites = np.random.randint(10, len(metabolites) + 1)
        chosen_metabolites = np.random.choice(metabolites, num_metabolites, replace=False)
        
        # Dictionary to store ground truth concentrations for this sample
        gt_row = {'Sample_ID': f'Sample_{i:04d}'}
        for meta in metabolites:
            gt_row[meta] = 0.0 # Initialize to 0
            
        for meta in chosen_metabolites:
            # Assign random concentration (using Gamma distribution similar to NMRQNet)
            concentration = np.random.gamma(shape=2.0, scale=1.0)
            # Add a small positional shift (alignment error)
            shift_error = np.random.normal(0, 0.0005)
            shift_idx = int(round(shift_error / 0.001))
            
            spec = pure_spectra[meta] * concentration
            
            # Apply shift
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
        
        data[f'Sample_{i:04d}'] = np.round(sample_intensity, 4)
        ground_truth.append(gt_row)
        
        if i % 200 == 0:
            print(f"Generated {i}/{num_samples} samples...")

    print("Converting to DataFrames...")
    df_data = pd.DataFrame(data)
    df_gt = pd.DataFrame(ground_truth)
    
    output_csv = os.path.join(output_dir, 'realistic_nmr_data_large.csv')
    gt_csv = os.path.join(output_dir, 'realistic_nmr_ground_truth.csv')
    
    print(f"Saving data to {output_csv}...")
    df_data.to_csv(output_csv, index=False)
    
    print(f"Saving ground truth to {gt_csv}...")
    df_gt.to_csv(gt_csv, index=False)
    
    print(f"Successfully generated {output_csv}")
    print(f"Successfully generated {gt_csv}")
    print(f"Data shape: {df_data.shape}")
    print(f"Ground Truth shape: {df_gt.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    args = parser.parse_args()
    
    generate_realistic_dataset(num_samples=args.samples, output_dir=args.output_dir)
