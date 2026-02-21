"""Shared library modules for 5-ALA fluorescence imaging pipeline.

Modules
-------
utils               Shared utilities: crop_ui, load_mask, load_analysis_mask, load_frame_data, etc.
blood_detection     Blood/dark-tissue detection using HSV + Otsu thresholding.
reflection_detection  Two-stage specular reflection detection.
fluorescence_transforms  Color-space transformations for fluorescence signal extraction.
topographic_analysis  Global topographic surface metrics.
polar_surface_analysis  Polar coordinate transform and radial/circumferential profiles.
persistent_homology  Sublevel-set persistent homology via cubical complexes.
"""
