"""HTML and PDF report generation utilities for suture analysis results.

This module provides functionality to generate HTML and PDF reports from suture analysis results,
including measurements, visualizations, and confidence scores.
"""

import os
import base64
from typing import Dict, List, Optional, Union
import pandas as pd
from IPython.display import HTML, display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
import json

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    if os.path.isfile(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    return "Image not found"

def generate_html_table(
    df: pd.DataFrame,
    img_width: int = 30,
    table_type: str = "suture"
) -> str:
    """Generate HTML table from DataFrame with embedded images.
    
    Args:
        df: DataFrame containing measurement data
        img_width: Width of embedded images in pixels
        table_type: Type of measurements to display
        
    Returns:
        HTML string of the generated table
    """
    headers_map = {
        "suture": ["Length of Suture (Pixels)", "Length of Suture (mm)"],
        "knot_incision": ["Length B/W Knot to Incision (Pixels)", "Length B/W Knot to Incision (mm)"],
        "tail_top": ["Length of Tail Top (Pixels)", "Length of Tail Top (mm)"],
        "tail_end": ["Length of Tail End (Pixels)", "Length of Tail End (mm)"],
    }
    
    extra_cols = headers_map.get(table_type, [])
    show_img = table_type != "knot_incision"
    
    # Generate header HTML
    header_html = "".join([
        f"<th>{col}</th>" for col in 
        ["Image Name", "Class Name", "Confidence"] + extra_cols + 
        (["Cropped Image"] if show_img else [])
    ])
    
    # Generate rows HTML
    rows_html = ""
    for _, row in df.iterrows():
        base = [row['Image Name'], row['Class Name'], f"{row['Confidence']:.2f}"]
        extra = [row.get(col, 'N/A') for col in extra_cols]
        img_html = (
            f"<img src='data:image/jpeg;base64,{encode_image_to_base64(row.get('Cropped Image', ''))}' "
            f"width='{img_width}' />"
            if show_img and os.path.isfile(row.get('Cropped Image', ''))
            else ("<b>Image not found</b>" if show_img else "")
        )
        all_cols = base + extra + ([img_html] if show_img else [])
        rows_html += "<tr>" + "".join([f"<td>{c}</td>" for c in all_cols]) + "</tr>"

    return (
        f"<table style='border-collapse: collapse; font-family: Arial; font-size: 14px;' border=1>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"
    )

def display_tables_by_image(
    df_suture: pd.DataFrame,
    df_knot: pd.DataFrame,
    df_top: pd.DataFrame,
    df_end: pd.DataFrame,
    img_width: int = 30
) -> None:
    """Display HTML tables organized by image.
    
    Args:
        df_suture: DataFrame with suture measurements
        df_knot: DataFrame with knot measurements
        df_top: DataFrame with tail top measurements
        df_end: DataFrame with tail end measurements
        img_width: Width of embedded images in pixels
    """
    html = ""
    for img in df_suture['Image Name'].unique():
        sections = [
            ("Suture Length", generate_html_table(
                df_suture[df_suture['Image Name'] == img], img_width, "suture"
            )),
            ("Knot-to-Incision Distance", generate_html_table(
                df_knot[df_knot['Image Name'] == img], img_width, "knot_incision"
            )),
            ("Tail Top Length", generate_html_table(
                df_top[df_top['Image Name'] == img], img_width, "tail_top"
            )),
            ("Tail End Length", generate_html_table(
                df_end[df_end['Image Name'] == img], img_width, "tail_end"
            )),
        ]
        html += (
            f"<h3>Suture Image: {img}</h3>" + 
            "".join([f"<h4>{title}</h4>{tbl}" for title, tbl in sections]) + 
            "<hr />"
        )
    display(HTML(html))

def generate_pdf_report(
    image_path: str,
    measurements: Dict[str, List[Dict]],
    output_path: str,
    title: str = "Suture Analysis Report"
) -> None:
    """Generate a comprehensive PDF report with measurements and visualizations.
    
    Args:
        image_path: Path to the input image
        measurements: Dictionary of measurement results
        output_path: Path to save the PDF report
        title: Title of the report
    """
    # Create PDF
    with PdfPages(output_path) as pdf:
        # Create figure
        fig = plt.figure(figsize=(11, 8.5))
        
        # Plot 1: Original image with measurements
        plt.subplot(2, 1, 1)
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot 2: Measurement statistics
        plt.subplot(2, 1, 2)
        plot_measurement_statistics(measurements)
        
        # Save page
        pdf.savefig(fig)
        plt.close()
        
        # Create summary page
        fig = plt.figure(figsize=(11, 8.5))
        plot_summary_statistics(measurements)
        pdf.savefig(fig)
        plt.close()

def plot_measurement_statistics(measurements: Dict[str, List[Dict]]) -> None:
    """Plot measurement statistics.
    
    Args:
        measurements: Dictionary of measurement results
    """
    # Group measurements by type
    grouped = {}
    for m_type, m_list in measurements.items():
        if m_list:
            values = [m['length_mm' if 'length_mm' in m else 'distance_mm'] for m in m_list]
            grouped[m_type] = values
    
    # Create box plots
    plt.boxplot([grouped[t] for t in grouped.keys()],
               labels=[t.replace('_', ' ').title() for t in grouped.keys()])
    plt.title('Measurement Statistics')
    plt.ylabel('Value (mm)')
    plt.xticks(rotation=45)

def plot_summary_statistics(measurements: Dict[str, List[Dict]]) -> None:
    """Plot summary statistics.
    
    Args:
        measurements: Dictionary of measurement results
    """
    # Calculate statistics
    stats = {}
    for m_type, m_list in measurements.items():
        if m_list:
            values = [m['length_mm' if 'length_mm' in m else 'distance_mm'] for m in m_list]
            stats[m_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': min(values),
                'max': max(values)
            }
    
    # Create table
    plt.table(cellText=[[f"{stats[t]['mean']:.2f} ± {stats[t]['std']:.2f}",
                        f"{stats[t]['min']:.2f}",
                        f"{stats[t]['max']:.2f}"]
                       for t in stats.keys()],
             colLabels=['Mean ± Std', 'Min', 'Max'],
             rowLabels=[t.replace('_', ' ').title() for t in stats.keys()],
             loc='center')
    
    plt.title('Measurement Summary')
    plt.axis('off')

def save_measurements_to_json(
    measurements: Dict[str, List[Dict]],
    output_path: str
) -> None:
    """Save measurements to JSON file.
    
    Args:
        measurements: Dictionary of measurement results
        output_path: Path to save the JSON file
    """
    data = {
        'timestamp': datetime.now().isoformat(),
        'measurements': measurements
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4) 