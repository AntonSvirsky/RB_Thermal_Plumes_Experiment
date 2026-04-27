# Rayleigh–Bénard Large Cell Thermal Plume Analysis

Python/OpenCV tool for processing **shadowgraph videos** recorded from a **Rayleigh–Bénard convection experiment in a large convection cell**.

In this setup, a fluid layer is heated from below and cooled from above. Above a critical temperature difference, buoyancy drives convective motion, producing rising hot plumes and descending cold plumes. These temperature-induced density variations modify the refractive index of the fluid, allowing the flow structures to be visualized using the **shadowgraph imaging method**.

This script analyzes such videos to extract plume motion, fluctuation patterns, and space-time dynamics using image processing, perspective correction, and cross-section analysis.

---

## What This Code Does

The script processes shadowgraph movies to isolate dynamic plume activity from the stationary background flow.

Main steps:

1. Correct camera perspective  
2. Compute time-averaged reference image  
3. Subtract mean flow background  
4. Enhance fluctuations  
5. Threshold plume structures  
6. Extract vertical cross-sections over time  
7. Build space-time diagrams for plume tracking  

---

## Typical Use Case

Designed for teaching laboratories. 

- Rayleigh–Bénard convection  
- Thermal plume tracking  
- Large-scale circulation visualization  
- Space-time analysis of coherent structures  

---

## Dependencies

Install required Python packages:

```bash
pip install opencv-python numpy matplotlib
```
## Folder Structure

```text
Movies/       # Input shadowgraph videos (.mp4)
Analysis/     # Generated outputs

Analysis/<video_name>/
├── orig_video.avi
├── flact_video.avi
├── reference frame.png
├── log.txt
├── cross-section imgs/
└── cross-section data/
```

## Author 
Anton Svirsky; 
Technion – Israel Institute of Technology;  
anton.sv@campus.technion.ac.il
