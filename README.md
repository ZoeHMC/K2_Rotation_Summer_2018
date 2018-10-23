# K2_Rotation_Summer_2018

Functions for Determining Rotation Periods of Stars from the K2 Data
====================================================================

These functions are intended to be used to determine a set of star rotation periods from the K2 data that are likely to be accurate by using both the Lomb-Scargle and autocorrelation methods and taking the subset of periods the two agree on within a certain threshold. They can also be used to produce various plots using this set of output periods, as well as plots that help illuminate the how the Lomb-Scargle or autocorrelation methods have been applied by the program to individual stars.

Make sure to check that all of the libraries that are imported at the top of FindP.py are installed and you have considered the root directory you want to be working within, as all of the functions require one to be set and it should usually be the same one across all of them.

Developed by: Zoë Bell and Dr. James R. A. Davenport
Funded by: NSF Astronomy and Astrophysics Postdoctoral Fellowship under award AST-1501418