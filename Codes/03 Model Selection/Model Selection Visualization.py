# -*- coding: utf-8 -*-
"""
GCM Selection Pipeline - Visualization (Step 5)
================================================
Produces 16 figures comparing ERA5 against CMIP6 GCMs at each selection
stage, split by SSP scenario (SSP245 and SSP585):

    Figure_1a : All models    | SSP245 | Precipitation
    Figure_1b : All models    | SSP245 | Temperature
    Figure_1c : All models    | SSP585 | Precipitation
    Figure_1d : All models    | SSP585 | Temperature

    Figure_2a : Top 24 models | SSP245 | Precipitation
    Figure_2b : Top 24 models | SSP245 | Temperature
    Figure_2c : Top 24 models | SSP585 | Precipitation
    Figure_2d : Top 24 models | SSP585 | Temperature

    Figure_3a : Top 16 models | SSP245 | Precipitation
    Figure_3b : Top 16 models | SSP245 | Temperature
    Figure_3c : Top 16 models | SSP585 | Precipitation
    Figure_3d : Top 16 models | SSP585 | Temperature

    Figure_4a : Final 4 models | SSP245 | Precipitation  (with legend)
    Figure_4b : Final 4 models | SSP245 | Temperature    (with legend)
    Figure_4c : Final 4 models | SSP585 | Precipitation  (with legend)
    Figure_4d : Final 4 models | SSP585 | Temperature    (with legend)

Notes:
    - All plots use the historical baseline monthly climatology (1985-2014)
    - ERA5 is always shown as a thick black reference line
    - Figures 1-3: no title, no legend
    - Figure 4: no title, legend shows the 4 final model names + ERA5
    - Run step1_to_4_computation.py first to generate required Excel files

Author: Arshad / Diwakar Adhikari
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# USER-DEFINED VARIABLES — Edit these before running
# =============================================================================
output_folder = r"C:\Users\Diwakar Adhikari\Downloads\climate data\Model Selection\01_selection\01_selection\MRB"
select_step1  = 24   # Number of models selected in Step 1
select_step2  = 16   # Number of models selected in Step 2

# =============================================================================
# CHECK REQUIRED INPUT FILES EXIST
# =============================================================================
required_files = [
    'Final_step_ssp245_corner_GCMs.xlsx',
    'Final_step_ssp585_corner_GCMs.xlsx',
    'ERA5_monthly_avg_pr_MRB.xlsx',
    'ERA5_monthly_avg_tas_MRB.xlsx',
    'GCMs_monthly_avg_pr_MRB.xlsx',
    'GCMs_monthly_avg_tas_MRB.xlsx',
    'step1_seasonal_bias_selected_GCMs.xlsx',
    'step2_annual_bias_selected_GCMs.xlsx',
]
for f in required_files:
    path = os.path.join(output_folder, f)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Please run step1_to_4_computation.py first."
        )
print("All required input files found. Starting visualization...\n")

# =============================================================================
# LOAD DATA
# =============================================================================
model_ssp245 = pd.read_excel(os.path.join(output_folder, 'Final_step_ssp245_corner_GCMs.xlsx'))
model_ssp585 = pd.read_excel(os.path.join(output_folder, 'Final_step_ssp585_corner_GCMs.xlsx'))
pr_ERA5      = pd.read_excel(os.path.join(output_folder, 'ERA5_monthly_avg_pr_MRB.xlsx'))
tas_ERA5     = pd.read_excel(os.path.join(output_folder, 'ERA5_monthly_avg_tas_MRB.xlsx'))
pr_GCMs      = pd.read_excel(os.path.join(output_folder, 'GCMs_monthly_avg_pr_MRB.xlsx'))
tas_GCMs     = pd.read_excel(os.path.join(output_folder, 'GCMs_monthly_avg_tas_MRB.xlsx'))

# Load saved model selection lists directly from the saved Excel files
step1_df = pd.read_excel(os.path.join(output_folder, 'step1_seasonal_bias_selected_GCMs.xlsx'))
step2_df = pd.read_excel(os.path.join(output_folder, 'step2_annual_bias_selected_GCMs.xlsx'))

# =============================================================================
# BUILD MODEL LISTS FOR EACH SELECTION STAGE
# =============================================================================

# All GCMs (from the GCMs Excel file)
all_gcm_models = pr_GCMs['Model'].tolist()

# Top 24: Step 1 selected + ERA5
model_selected_step1 = ['ERA5'] + step1_df['GCMs'].tolist()

# Top 16: Step 2 selected + ERA5
model_selected_step2 = ['ERA5'] + step2_df['GCMs'].tolist()

# Final 4 for SSP245 (CD, CW, WD, WW corner models)
final_models_ssp245 = model_ssp245.values.flatten().tolist()

# Final 4 for SSP585
final_models_ssp585 = model_ssp585.values.flatten().tolist()

# =============================================================================
# PREPARE MONTHLY CLIMATOLOGY DATA IN LONG FORMAT FOR PLOTTING
# =============================================================================
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Combine ERA5 + all GCMs
df_ppt  = pd.concat([pr_ERA5,  pr_GCMs],  ignore_index=True)
df_temp = pd.concat([tas_ERA5, tas_GCMs], ignore_index=True)

# Melt to long format
ppt_monthly  = pd.melt(df_ppt,  id_vars=['Model'], value_vars=month_order,
                        var_name='Month', value_name='PPT')
temp_monthly = pd.melt(df_temp, id_vars=['Model'], value_vars=month_order,
                        var_name='Month', value_name='T')

# Enforce correct month order on x-axis
ppt_monthly['Month']  = pd.Categorical(ppt_monthly['Month'],
                                        categories=month_order, ordered=True)
temp_monthly['Month'] = pd.Categorical(temp_monthly['Month'],
                                        categories=month_order, ordered=True)

# =============================================================================
# HELPER: Plot function
# =============================================================================
def make_plot(data, y_col, ylabel, save_path, show_legend=False, legend_models=None, title=None):
    """
    Draw a monthly climate line plot and save as PNG.

    Parameters
    ----------
    data          : DataFrame in long format with columns [Model, Month, y_col]
    y_col         : 'PPT' or 'T'
    ylabel        : y-axis label string
    save_path     : full path to save the PNG
    show_legend   : if True, show legend (used for Figure 4 only)
    legend_models : list of the 4 final model names to include in legend
    """
    unique_models = data['Model'].unique()

    # Build colour and linewidth maps; ERA5 always black and thicker
    palette   = {model: 'black' if model == 'ERA5'
                 else sns.color_palette("tab10", len(unique_models))[i]
                 for i, model in enumerate(unique_models)}
    linewidth = {model: 3 if model == 'ERA5' else 2 for model in unique_models}

    # Plot ERA5 last so it renders on top of all GCM lines
    plot_order = [m for m in unique_models if m != 'ERA5']
    if 'ERA5' in unique_models:
        plot_order.append('ERA5')

    fig, ax = plt.subplots(figsize=(17, 14))

    for model in plot_order:
        subset = data[data['Model'] == model].sort_values('Month')
        sns.lineplot(
            data=subset, x='Month', y=y_col,
            label=model, ax=ax,
            linewidth=linewidth[model],
            color=palette[model]
        )

    # Title only shown if provided (Figure 4 only)
    ax.set_title(title if title else '', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel('Months', fontsize=16)
    ax.grid(True, color='r', dashes=(5, 2, 1, 2))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    if show_legend and legend_models:
        # Show only the final model names + ERA5 in the legend
        handles, labels = ax.get_legend_handles_labels()
        keep = legend_models + ['ERA5']
        filtered = [(h, l) for h, l in zip(handles, labels) if l in keep]
        if filtered:
            fh, fl = zip(*filtered)
            ax.legend(fh, fl, loc='upper center',
                      bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)
        else:
            ax.get_legend().remove()
    else:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=1000)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# =============================================================================
# FIGURE 1: ALL MODELS — SSP245 (1a, 1b) and SSP585 (1c, 1d)
# =============================================================================
print("=" * 60)
print("Generating Figure 1 — All Models")
print("=" * 60)

all_ppt  = ppt_monthly[ppt_monthly['Model'].isin(['ERA5'] + all_gcm_models)]
all_temp = temp_monthly[temp_monthly['Model'].isin(['ERA5'] + all_gcm_models)]

make_plot(all_ppt,  'PPT', 'Precipitation (mm)',
          os.path.join(output_folder, 'Figure_1a_All_Models_SSP245_Precipitation.png'))

make_plot(all_temp, 'T',   'Temperature (°C)',
          os.path.join(output_folder, 'Figure_1b_All_Models_SSP245_Temperature.png'))

make_plot(all_ppt,  'PPT', 'Precipitation (mm)',
          os.path.join(output_folder, 'Figure_1c_All_Models_SSP585_Precipitation.png'))

make_plot(all_temp, 'T',   'Temperature (°C)',
          os.path.join(output_folder, 'Figure_1d_All_Models_SSP585_Temperature.png'))


# =============================================================================
# FIGURE 2: TOP 24 MODELS — SSP245 (2a, 2b) and SSP585 (2c, 2d)
# =============================================================================
print("=" * 60)
print("Generating Figure 2 — Top 24 Models (Step 1 Selected)")
print("=" * 60)

step1_ppt  = ppt_monthly[ppt_monthly['Model'].isin(model_selected_step1)]
step1_temp = temp_monthly[temp_monthly['Model'].isin(model_selected_step1)]

make_plot(step1_ppt,  'PPT', 'Precipitation (mm)',
          os.path.join(output_folder, 'Figure_2a_Top24_SSP245_Precipitation.png'))

make_plot(step1_temp, 'T',   'Temperature (°C)',
          os.path.join(output_folder, 'Figure_2b_Top24_SSP245_Temperature.png'))

make_plot(step1_ppt,  'PPT', 'Precipitation (mm)',
          os.path.join(output_folder, 'Figure_2c_Top24_SSP585_Precipitation.png'))

make_plot(step1_temp, 'T',   'Temperature (°C)',
          os.path.join(output_folder, 'Figure_2d_Top24_SSP585_Temperature.png'))


# =============================================================================
# FIGURE 3: TOP 16 MODELS — SSP245 (3a, 3b) and SSP585 (3c, 3d)
# =============================================================================
print("=" * 60)
print("Generating Figure 3 — Top 16 Models (Step 2 Selected)")
print("=" * 60)

step2_ppt  = ppt_monthly[ppt_monthly['Model'].isin(model_selected_step2)]
step2_temp = temp_monthly[temp_monthly['Model'].isin(model_selected_step2)]

make_plot(step2_ppt,  'PPT', 'Precipitation (mm)',
          os.path.join(output_folder, 'Figure_3a_Top16_SSP245_Precipitation.png'))

make_plot(step2_temp, 'T',   'Temperature (°C)',
          os.path.join(output_folder, 'Figure_3b_Top16_SSP245_Temperature.png'))

make_plot(step2_ppt,  'PPT', 'Precipitation (mm)',
          os.path.join(output_folder, 'Figure_3c_Top16_SSP585_Precipitation.png'))

make_plot(step2_temp, 'T',   'Temperature (°C)',
          os.path.join(output_folder, 'Figure_3d_Top16_SSP585_Temperature.png'))


# =============================================================================
# FIGURE 4: FINAL 4 MODELS — SSP245 (4a, 4b) and SSP585 (4c, 4d)
# Legend shows the 4 final model names + ERA5
# =============================================================================
print("=" * 60)
print("Generating Figure 4 — Final 4 Models")
print("=" * 60)

# SSP245 final 4
ssp245_ppt  = ppt_monthly[ppt_monthly['Model'].isin(['ERA5'] + final_models_ssp245)]
ssp245_temp = temp_monthly[temp_monthly['Model'].isin(['ERA5'] + final_models_ssp245)]

make_plot(ssp245_ppt,  'PPT', 'Precipitation (mm)',
          os.path.join(output_folder, 'Figure_4a_Final4_SSP245_Precipitation.png'),
          show_legend=True, legend_models=final_models_ssp245,
          title='Top 4 Models SSP245 Precipitation')

make_plot(ssp245_temp, 'T',   'Temperature (°C)',
          os.path.join(output_folder, 'Figure_4b_Final4_SSP245_Temperature.png'),
          show_legend=True, legend_models=final_models_ssp245,
          title='Top 4 Models SSP245 Temperature')

# SSP585 final 4
ssp585_ppt  = ppt_monthly[ppt_monthly['Model'].isin(['ERA5'] + final_models_ssp585)]
ssp585_temp = temp_monthly[temp_monthly['Model'].isin(['ERA5'] + final_models_ssp585)]

make_plot(ssp585_ppt,  'PPT', 'Precipitation (mm)',
          os.path.join(output_folder, 'Figure_4c_Final4_SSP585_Precipitation.png'),
          show_legend=True, legend_models=final_models_ssp585,
          title='Top 4 Models SSP585 Precipitation')

make_plot(ssp585_temp, 'T',   'Temperature (°C)',
          os.path.join(output_folder, 'Figure_4d_Final4_SSP585_Temperature.png'),
          show_legend=True, legend_models=final_models_ssp585,
          title='Top 4 Models SSP585 Temperature')


print("\n" + "=" * 60)
print("ALL 16 FIGURES COMPLETE.")
print("Saved to:", output_folder)
print("=" * 60)
