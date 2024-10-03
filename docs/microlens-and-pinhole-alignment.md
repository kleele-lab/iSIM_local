# Alignment of the iSIM Microlens and Pinhole Arrays

The new version of the protocol has been updated to reflect how the most recent alignments had been completed successfully. It is not strictly better or worse than version 1.0, might be somewhat simpler.

This SOP assumes all the alignment steps up to 2.2.1.5 have been completed (based on [Curd et al., Methods 2015](https://doi.org/10.1016/j.ymeth.2015.07.012)). For the alignment, you should prepare the following samples:

1. Concentrated dye solution: This sample is prepared based on the [Andor technical note](https://andor.oxinst.com/learning/view/article/how-flat-is-your-confocal-illumination-profile). For the iSIM, the NaFITC dye should be used for the 488
nm laser.
2. Dense 100 nm bead sample: Sonicate the 100 nm bead solution and place directly a few uL on a PLL treated coverslip. Leave to dry.
3. Sparse 100 nm bead sample: Dilute the 100 nm bead solution in water at 1/100 or 1/1000. Put a 10-30 uL on a PLL treated coverslip and leave to dry.

All coverslips should be #1.5 (0.17 mm) thickness, preferably plasma cleaned and treated with PLL.

Caution: There are two objectives available on the iSIM (TIRFAPON60XO and PLANAPO60XO). If using the TIRF objective, make sure to place the correction collar to 28 ºC for the alignment, and this is recommended for all imaging (both at 28 ºC and 37 ºC). If you ever wish to align for 37 ºC, this will require readjusting the position of the tube lens, which is coupled to the position of scan lens 2, etc.
At the beginning of the alignment, take all of the microarray elements out.

## Excitation path

### Start the alignment software

There is an alignment GUI that facilitates the alignment of the pinholes. To launch the alignment tool, first start Micro-Manager. In the `pymm-eventserver` plugin GUI window, ensure that the `Live Mode Events` checkbox is checked.

Next, start the alignment GUI from the GUI code directory (`C:\iSIM\isimgui` at the time of this writing) with the following command:

```console
py main.py alignment
```

### Adjust EXML axial position:

Use the NaFITC dye lake sample and find the focus in widefield mode. To focus, use structured features within the dye lake, such as little crystals or edges of bubbles that allow you to focus better. Make sure the sample is properly focused before introducing the EXML array. Adjust the z (axial) position of the EXML array so that the excitation spots appear most in focus on the camera. As readout for the spot "focus", I suggest using the dynamic range of the camera and visualizing the intensity distribution in log form. Changing the z position will cause the histogram to move towards higher values as you approach the focus, and decrease further out. Adjust this iteratively, while checking that the sample is still in focus in widefield mode.

### Adjust EXML lateral position and orientation:

Once the spots are in focus, the z position should not be changed anymore. Now, the goal is to match the orientation and lateral position of the EXML to a reference. Still using the dye lake sample, load one of the alignment lines (most recent alignment line 3) to use as a reference. First, translate the EXML in x and y (laterally), to position one of the excitation spots on the center of the line. Next, check if the orientation of the EXML is correct by observing whether the other points along the alignment line are correctly positioned. If not, rotate the EXML, recenter the central spot and check again. This procedure has to be done iteratively (adjusting orientation and recentering), since rotating the EXML changes the lateral position as well. Once the orientation is correct, adjust only the lateral position of the EXML to position an excitation spot on the center of the line. To adjust along the other dimension, use a transparent ruler to check the position of the central spot with respect to the neighboring spots. Once the lateral position has been adjusted, the EXML should not be touched.

### Emission path

### Adjust EMML lateral position and orientation

The dye sample is no longer needed since the emission path can be adjusted from brightfield light by shining a lamp or light from your phone from the objective side. Put in the EMML and adjust its orientation and lateral position using a similar procedure as above. In this step, the orientation is more important to get correctly, since it will not be possible to correct for later, while the lateral position should be close to the reference line but will require later re-adjustment. Briefly, using the transmitted light through the EMML and the reference line, place one of the spots on the center of the line and check whether the points extending outwards are aligned to the reference line. If not, rotate the EMML array and verify the orientation again by re-positioning a central spot on the center of the line and checking the orientation of the neighboring spots. Iterate until the orientation does not require further adjustment. Once the orientation of the EMML is correct, adjust the lateral position to roughly place a single spot on the center of the line. At the end, gently take out the EMML.

### Adjust EMPH lateral position and orientation
The EMPH array axial position is aligned during the construction procedure and should not require adjustment after. Instead, to correct the lateral position and orientation, use a similar procedure as described above. Briefly, place a single pinhole at the center of the reference line. Check the orientation of the EMPH by comparing the positions of the neighboring points to the reference line. If their positions are not on the line, rotate the EMPH and recenter the central spot. Iterate the angle verification and recentering until the EMPH orientation does not require further adjustment. Then, adjust the x and y positions of the EMPH array to center a spot at the center of the line. Once aligned, the EMPH should not be perturbed further.

### Adjust EMML axial position

Gently re-introduce the EMML array. The orientation should not require further adjustment. Instead, adjust the z (axial) position of the EMML to focus the images of the pinholes as much as possible. Similarly to step 1., refer to the log scaled intensity histogram as a readout for the focus of the spots, while visually verifying the spot shape is circular and contracted. Once the focus has been achieved, re-adjust the lateral position of the EMML only slightly to match better to the alignment line center. verify the spots are still in focus and circular. Repeat until no further improvement. Verify alignment:

### Check light transmission

If the dye sample is still mounted, try running the microscope in iSIM mode and observing whether the features used to focus in step 1 are observable. If not, or if very dim, it is likely there is a mismatch between
the array elements, blocking full light transmission.

### Check resolution:

Mount the sparse bead sample and find the focus in iSIM mode. Take a 3D stack of beads and measure the FWHM profile of a bead when in full focus around the center of the image. If well aligned, the system should give a resolution of ~213 nm FWHM before deconvolution, corresponding to a sigma value of ~90 nm. If the measured resolution is insufficient, it is likely that either the excitation is not properly focused on the sample or that the EMML is not contracting the images of the pinholes fully. If you think it is more likely to be the latter, you can try adjusting just the EMML from step 5, however, if you are not sure of the source of the problem, repeat from step 1.
